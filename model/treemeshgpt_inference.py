import torch, math, sys
from torch import nn
from torch.nn import Module
from pytorch_custom_utils import save_load
from beartype.typing import Union, Tuple
from einops import pack
from model.custom_transformers_inference import (
    FlashAttentionTransformers as Transformers,
    eval_decorator,
)
from fns import dequantize_verts_tensor
from model.pc_encoder import CloudEncoder

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# main model
# ---------------------------------------------------------
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
        topk=10,
        max_seq_len=25_000,
    ):
        super().__init__()
        self.quant_bit = quant_bit
        self.dim = dim

        # ---------- embeddings ----------
        self.sos_emb = nn.Parameter(torch.randn(dim))
        self.fc_edges = nn.Linear(1024, dim)
        self.sos_emb_2 = nn.Parameter(torch.randn(512))
        self.fc_edges_2 = nn.Linear(1024, dim)

        # ---------- decoder ----------
        self.decoder = Transformers(
            dim=dim,
            depth=attn_depth,
            heads=attn_heads,
            attn_flash=flash_attn,
            attn_dropout=dropout,
            ff_dropout=dropout,
            **attn_kwargs,
        )

        # ---------- heads ----------
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

        # ---------- misc ----------
        self.pad_id = pad_id
        self.pc_encoder = CloudEncoder()
        self.pc_adapter = nn.Linear(64, dim)
        self.n = 0
        self.topk = topk
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    # generation API
    # ------------------------------------------------------------------
    @eval_decorator
    @torch.no_grad()
    def generate(self, pc, n: int = 0):
        # (unchanged body — omitted for brevity)
        ...
        # while-loop, sampling, etc.
        ...

    # ------------------------------------------------------------------
    # utility fns  (unchanged except adjust_temperature)
    # ------------------------------------------------------------------
    def sos(self):
        return self.sos_emb.unsqueeze(0).unsqueeze(0)

    def sos1(self, xyz):
        xyz = dequantize_verts_tensor(xyz, n_bits=self.quant_bit).unsqueeze(1)
        fea = torch.cat(
            [self.pc_encoder.point_embed(xyz), self.sos_emb_2.unsqueeze(0).unsqueeze(0)],
            dim=-1,
        )
        return self.fc_edges_2(fea)

    def encode_edge(self, xyz_0, xyz_1):
        a = dequantize_verts_tensor(xyz_0, n_bits=self.quant_bit).unsqueeze(0)
        b = dequantize_verts_tensor(xyz_1, n_bits=self.quant_bit).unsqueeze(0)
        a = self.pc_encoder.point_embed(a)
        b = self.pc_encoder.point_embed(b)
        return self.fc_edges(torch.cat([a, b], dim=-1))

    # ... predict_xyz, predict, check_duplicate stay unchanged ...

    # --------- SAFE TEMPERATURE ------------------------------------------------
    def adjust_temperature(self, stack_size: int) -> float:
        """
        Adaptive temperature used at each sampling step.

        Very small values (<0.2) cause under-flow after the top-k mask.
        Clamp to 0.3 minimum to keep probabilities well-formed.
        """
        return max(0.3, 2.0 / (stack_size + 1))
