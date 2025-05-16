import math, sys, random, numpy as np
import torch
from torch import nn
from torch.nn import Module
from beartype.typing import Union, Tuple
from einops import pack

from pytorch_custom_utils import save_load
from model.custom_transformers_inference import (
    FlashAttentionTransformers as Transformers,
    eval_decorator,
)
from model.pc_encoder import CloudEncoder
from fns import dequantize_verts_tensor


# ─── helpers ──────────────────────────────────────────────────────────
def get_positional_encoding(L, D, device="cpu"):
    position = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, D, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / D)
    )
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ─── model ────────────────────────────────────────────────────────────
@save_load()
class TreeMeshGPT(Module):
    def __init__(
        self,
        *,
        dim: Union[int, Tuple[int, int]] = 1024,
        flash_attn=True,
        attn_depth=24,
        attn_heads=16,
        attn_kwargs: dict = dict(ff_glu=True),
        dropout=0.0,
        quant_bit=7,
        pad_id=-1,
        max_seq_len=30000,
    ):
        super().__init__()

        self.quant_bit = quant_bit
        self.dim = dim

        # embeddings -----------------------------------------------------------------
        self.sos_emb = nn.Parameter(torch.randn(dim))
        self.fc_edges = nn.Linear(1024, dim)
        self.sos_emb_2 = nn.Parameter(torch.randn(512))
        self.fc_edges_2 = nn.Linear(1024, dim)

        # transformer ----------------------------------------------------------------
        self.decoder = Transformers(
            dim=dim,
            depth=attn_depth,
            heads=attn_heads,
            attn_flash=flash_attn,
            attn_dropout=dropout,
            ff_dropout=dropout,
            **attn_kwargs,
        )

        # heads ----------------------------------------------------------------------
        self.head_coord1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2**quant_bit + 2),
        )

        self.coord1_emb = nn.Embedding(2**quant_bit, dim)
        self.head_coord2 = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2**quant_bit),
        )

        self.coord2_emb = nn.Embedding(2**quant_bit, dim)
        self.head_coord3 = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2**quant_bit),
        )

        # misc -----------------------------------------------------------------------
        self.pad_id = pad_id
        self.pc_encoder = CloudEncoder()
        self.pc_adapter = nn.Linear(64, dim)
        self.n = 0                      # bias for EOS trick
        self.topk = 1                   # ← deterministic arg-max
        self.max_seq_len = max_seq_len

    # ─── generation loop ──────────────────────────────────────────────
    @eval_decorator
    @torch.no_grad()
    def generate(self, pc, n: int = 0):
        # deterministic RNG ----------------------------------------------------------
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        device = self.sos_emb.device
        self.n = -n

        # helpers -------------------------------------------------------------------
        def pe(idx):
            return init_pe[:, idx][:, None]

        def add_stack(e):
            stack.append({"edges": e})

        # initial tensors -----------------------------------------------------------
        dim = self.dim
        pad = torch.tensor([[-1, -1, -1]], device=device)
        edge_pad = torch.cat([pad, pad], dim=-1)
        edges = torch.empty((0, 6), device=device).long()
        pred = torch.empty((1, 0, 3), device=device).long()
        init_pe = get_positional_encoding(30000, 1024, device=device).unsqueeze(0)
        acc_fea = torch.empty((1, 0, dim), device=device)

        # encode point cloud ---------------------------------------------------------
        pc_embed = self.pc_encoder(pc.float())
        acc_fea = pack([acc_fea, self.pc_adapter(pc_embed)], "b * d")[0]
        _, cache = self.decoder(acc_fea, return_hiddens=True)

        # generation state -----------------------------------------------------------
        p = 0
        eos = False
        first = True
        stack = []

        # helper to add first three vertices of a component --------------------------
        def init_component(edges_, acc_fea_, pred_, p_, cache_, first_):
            # v0
            fea = self.sos() + pe(p_)
            acc_fea_ = pack([acc_fea_, fea], "b * d")[0]
            v0, eos_, cache_ = self.predict(acc_fea_, t=0.5, first=first_, kv_cache=cache_)
            pred_ = pack([pred_, v0], "b * d")[0]
            p_ += 1
            if eos_:
                return edges_, acc_fea_, pred_, p_, cache_, eos_, first_

            edges_ = torch.cat([edges_, torch.cat([v0, pad], dim=-1)], dim=0)

            # v1
            fea = self.sos1(v0) + pe(p_)
            acc_fea_ = pack([acc_fea_, fea], "b * d")[0]
            v1, eos_, cache_ = self.predict(acc_fea_, t=0.5, first=first_, kv_cache=cache_)
            pred_ = pack([pred_, v1], "b * d")[0]
            p_ += 1
            if eos_:
                return edges_, acc_fea_, pred_, p_, cache_, eos_, first_

            first_ = False
            edges_ = torch.cat([edges_, torch.cat([v0, v1], dim=-1)], dim=0)
            add_stack([v0, v1])

            # v2
            fea = self.encode_edge(v0, v1) + pe(p_)
            acc_fea_ = pack([acc_fea_, fea], "b * d")[0]
            v2, eos_, cache_ = self.predict(acc_fea_, t=0.5, kv_cache=cache_)
            pred_ = pack([pred_, v2], "b * d")[0]
            p_ += 1
            if eos_:
                return edges_, acc_fea_, pred_, p_, cache_, eos_, first_

            add_stack([v2, v0])
            add_stack([v1, v2])
            return edges_, acc_fea_, pred_, p_, cache_, eos_, first_

        # main loop ------------------------------------------------------------------
        while not eos and pred.shape[1] < self.max_seq_len:
            edges = torch.cat([edges, edge_pad], dim=0)
            edges, acc_fea, pred, p, cache, eos, first = init_component(
                edges, acc_fea, pred, p, cache, first
            )
            if eos:
                break

            while stack and pred.shape[1] < self.max_seq_len:
                cur = stack.pop()
                cur_edges = torch.cat([cur["edges"][1], cur["edges"][0]], dim=-1)

                prev_faces = (
                    torch.cat([edges.unsqueeze(0), pred], dim=-1).reshape(-1, 3, 3)
                )
                face_mask = (prev_faces != -1).all(dim=(1, 2))
                prev_faces = prev_faces[face_mask]

                edges = torch.cat([edges, cur_edges], dim=0)
                fea = self.encode_edge(cur["edges"][1], cur["edges"][0]) + pe(p)
                acc_fea = pack([acc_fea, fea], "b * d")[0]

                xyz_res, eos, cache = self.predict(acc_fea, t=0.5, kv_cache=cache)

                if xyz_res.sum() != -3:
                    cur_face = (
                        torch.cat([cur_edges, xyz_res], dim=-1).reshape(-1, 3, 3)[0]
                    )
                    if self.check_duplicate(prev_faces, cur_face):
                        xyz_res = torch.tensor([-1, -1, -1], device=fea.device).unsqueeze(0)

                print(
                    f"\rSequence length: {pred.shape[1]}/{self.max_seq_len} | "
                    f"Stack length: {len(stack):<4}",
                    end="",
                    flush=True,
                )

                pred = pack([pred, xyz_res], "b * d")[0]
                p += 1

                if xyz_res.sum() not in (-3, -6):
                    add_stack([xyz_res, cur["edges"][1]])
                    add_stack([cur["edges"][0], xyz_res])

                if eos:
                    break

        mask = ~((pred[0] < 0).any(dim=-1) | (edges < 0).any(dim=-1))
        triangles = torch.cat([edges[mask], pred[0][mask]], dim=-1).reshape(-1, 3, 3)
        triangles = dequantize_verts_tensor(triangles, n_bits=self.quant_bit)
        return triangles

    # ─── helpers identical to upstream (unchanged) ────────────────────
    def sos(self):
        return self.sos_emb.unsqueeze(0).unsqueeze(0)

    def sos1(self, xyz):
        xyz = dequantize_verts_tensor(xyz, n_bits=self.quant_bit).unsqueeze(1)
        fea = torch.cat(
            [self.pc_encoder.point_embed(xyz), self.sos_emb_2.unsqueeze(0).unsqueeze(0)],
            dim=-1,
        )
        return self.fc_edges_2(fea)

    def encode_edge(self, xyz0, xyz1):
        a = dequantize_verts_tensor(xyz0, n_bits=self.quant_bit).unsqueeze(0)
        b = dequantize_verts_tensor(xyz1, n_bits=self.quant_bit).unsqueeze(0)
        return self.fc_edges(torch.cat([self.pc_encoder.point_embed(a), self.pc_encoder.point_embed(b)], dim=-1))

    # deterministic arg-max sampling ----------------------------------
    def predict_xyz(
        self, res, dequantize=False, top_k=1, temperature=0.5, init_mask=False, first=False
    ):
        logits_z = self.head_coord1(res) / temperature
        logits_z[0][-1] += self.n
        if init_mask:
            logits_z[0][-2] = -999
        if first:
            logits_z[0][-2:] = -999

        z = logits_z.argmax(dim=-1)  # ← pick best token
        eos = False

        if z < 2 ** self.quant_bit:
            emb_z = self.coord1_emb(z)
            logits_y = self.head_coord2(torch.cat([res, emb_z], dim=-1)) / temperature
            y = logits_y.argmax(dim=-1)

            emb_y = self.coord2_emb(y)
            logits_x = self.head_coord3(torch.cat([res, emb_z, emb_y], dim=-1)) / temperature
            x = logits_x.argmax(dim=-1)

            xyz = torch.cat([x, y, z], dim=-1)
            if dequantize:
                xyz = dequantize_verts_tensor(xyz, n_bits=self.quant_bit)

        elif z == 2 ** self.quant_bit:
            xyz = torch.tensor([-1, -1, -1], device=z.device)
        else:
            xyz = torch.tensor([-2, -2, -2], device=z.device)
            eos = True

        return xyz, eos

    def predict(self, acc_fea, t=0.5, init_mask=False, first=False, kv_cache=None):
        res, cache = self.decoder(acc_fea, cache=kv_cache, return_hiddens=True)
        res = res[0]
        xyz, eos = self.predict_xyz(res, temperature=t, init_mask=init_mask, first=first)
        return xyz.unsqueeze(0), eos, cache

    def check_duplicate(self, prev_faces, cur_face):
        rotated = torch.cat(
            [prev_faces, prev_faces[:, [1, 2, 0]], prev_faces[:, [2, 0, 1]]], dim=0
        )
        return (rotated == cur_face).all(dim=(1, 2)).any()

    # constant safe temperature ---------------------------------------
    def adjust_temperature(self, stack_size: int):
        return 0.2
