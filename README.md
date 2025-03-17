

<br>
<p align="center">
<h1 align="center"><strong>🌲 TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing</strong></h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?user=w6RfcvMAAAAJ&hl=en' target='_blank'>Stefan Lionar</a>&emsp;
    <a href='https://jiabinliang.github.io/' target='_blank'>Jiabin Liang</a>&emsp;
    <a href='https://scholar.google.ca/citations?user=7hNKrPsAAAAJ&hl=en' target='_blank'>Gim Hee Lee</a>&emsp;
    <br>
    Sea AI Lab&emsp;Garena&emsp;National University of Singapore
    <h2 align="center">CVPR 2025</h2>
  </p>
</p>


<div align="center">
<a href="https://arxiv.org/abs/2503.11629"><img src="https://img.shields.io/badge/arXiv-2503.11629-blue?"></a> &nbsp;&nbsp;
 <a href="https://arxiv.org/pdf/2503.11629"><img src="https://img.shields.io/badge/Paper-📖-blue?"></a> &nbsp;&nbsp;
<a href="https://colab.research.google.com/drive/1UuYwl_GzkVmvcSReyqueMpOIsqr2u3cG?usp=sharing"><img src="https://img.shields.io/badge/Demo-Colab-F9AB00?logo=googlecolab&logoColor=yellow"></a>
</div>

## 💡 About
<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: justify;">
    <img src="assets/teaser-github.png" alt="Dialogue_Teaser" width=100% >


This is the **official repository** of **🌲 TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing**.  

TreeMeshGPT is an autoregressive Transformer designed to generate high-quality artistic meshes from input point clouds. Unlike conventional autoregressive models that rely on next-token prediction, TreeMeshGPT retrieves the next token from a dynamically growing tree structure, enabling localized mesh extensions and enhanced generation quality. Our novel Autoregressive Tree Sequencing method introduces an efficient face tokenization strategy, achieving a 22% compression rate compared to naive tokenization. This approach reduces training difficulty while allowing for the generation of meshes with finer details and consistent normal orientation. With 7-bit discretization, TreeMeshGPT supports meshes with up to 5,500 faces, while the 9-bit model extends this capability to 11,000 faces.
</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#-code-availability">Code availability</a>
    </li>
    <li>
      <a href="#-getting-started">Getting started</a>
    </li>
    <li>
      <a href="#-inference">Inference</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>


## 📌 Code Availability
- [x] [Google Colab demo](https://colab.research.google.com/drive/1UuYwl_GzkVmvcSReyqueMpOIsqr2u3cG?usp=sharing) – Run TreeMeshGPT in your browser.
- [x] Inference - Generate artistic mesh conditioned on point cloud sampled from dense mesh (`inference.py`)
- [x] Tokenizer - Create input-output pair for Autoregressive Tree Sequencing (`tokenizer.py`)
- [ ] Training script
- [ ] Dataset


## 🚀 Getting Started

To set up **TreeMeshGPT**, follow the steps below:

### 1. Clone the repository and create conda environment

```
git clone https://github.com/sail-sg/TreeMeshGPT.git
cd TreeMeshGPT

conda create -n tmgpt python=3.11
conda activate tmgpt
```
### 2. Install PyTorch≥2.5.0 with CUDA support

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
PyTorch≥2.5.0 is needed as we use FlexAttention for the point cloud condition during training. Older versions should work for inference. 

### 3. Install additional dependencies

```
pip install -r requirements.txt
```

### 4.Download pre-trained models

Download and arrange the pre-trained models using the following commands:

```
mkdir checkpoints

# 7-bit model
gdown 1IERM76szBQq9oAMoFw1Sbgp3kx97_Kib
mv treemeshgpt_7bit.pt checkpoints/

# 9-bit model
gdown 19Auv48x7kgoODRS7dij8QijWzDqRjq97
mv treemeshgpt_9bit.pt checkpoints/
```

Alternatively, you can download them manually from the following links and put them inside `checkpoints` folder:

- **[7-bit Model (treemeshgpt_7bit.pt)](https://drive.google.com/uc?id=1IERM76szBQq9oAMoFw1Sbgp3kx97_Kib)**
- **[9-bit Model (treemeshgpt_9bit.pt)](https://drive.google.com/uc?id=19Auv48x7kgoODRS7dij8QijWzDqRjq97)**

## 🎨 Inference

We provide a demo of **artistic mesh generation** conditioned on a point cloud sampled from a dense mesh. The dense meshes are generated using the **text-to-3D model from [Luma AI](https://lumalabs.ai/genie)**. Our demo can also be run on [Google Colab](https://colab.research.google.com/drive/1UuYwl_GzkVmvcSReyqueMpOIsqr2u3cG?usp=sharing).

### 🔹 **Run the Inference Script**
Use the following command to generate a mesh:
```bash
python inference.py
```

The output will be saved to `generation` folder.

### 🔹 Arguments Summary

| **Argument**                     | **Type**  | **Default**                                       | **Description** |
|-----------------------------------|----------|-------------------------------------------------|----------------|
| `--version`                      | `str`    | `7bit`                                        | Select model version: `7bit` or `9bit`. |
| `--ckpt_path`                    | `str`    | `./checkpoints/treemeshgpt_7bit.pt` (7-bit) / `./checkpoints/treemeshgpt_9bit.pt` (9-bit) | Path to the model checkpoint. |
| `--mesh_path`                     | `str`    | `demo/luma_cat.glb`                           | Path to the input mesh file. |
| `--decimation`                    | `bool`   | `True`| Enable or disable mesh decimation. Recommendation: `True` if input is dense mesh and `False` if input is Objaverse mesh.|
| `--decimation_target_nfaces`      | `int`    | `6000`                                          | Target number of faces after decimation. Use smaller number if generated mesh contains too many small triangles.|
| `--decimation_boundary_deletion`  | `bool`   | `True` (7-bit) / `False` (9-bit)                | Allow deletion of boundary vertices of decimated mesh. Set to `True` if generated mesh contains too many small triangles.|
| `--sampling`                      | `str`    | `uniform` (7-bit) / `fps` (9-bit)           | Sampling method: `uniform` if 7-bit and `fps` if 9-bit. |

### 🔹 Other example usages

- Run with 9-bit model and a mesh specified in `--mesh_path`:

    ```bash
    python inference.py --version 9bit --mesh_path demo/luma_bunny.glb
    ```

- Set `--decimation_boundary_deletion` to `True` and optionally use a lower `--decimation_target_nfaces` if the default configuration results in meshes with too many small triangles:

    ```bash
    python inference.py --version 9bit --mesh_path demo/luma_box.glb --decimation_boundary_deletion True --decimation_target_nfaces 2000
    ```

- Run without decimation (e.g., for Objaverse evaluation):

    ```bash
    python inference.py --decimation False --mesh_path demo/objaverse_pig.obj
    ```


## Acknowledgement
Our code is built on top of [PivotMesh](https://github.com/whaohan/pivotmesh) codebase. Our work is also inspired by these projects:
- [EdgeRunner](https://research.nvidia.com/labs/dir/edgerunner/)  
- [MeshAnything](https://buaacyw.github.io/mesh-anything/)  
- [MeshAnythingV2](https://buaacyw.github.io/meshanything-v2/)  
- [MeshXL](https://meshxl.github.io/)  
- [MeshGPT](https://nihalsid.github.io/mesh-gpt/)  
