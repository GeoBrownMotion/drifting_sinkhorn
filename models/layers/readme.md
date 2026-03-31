Basic layers for building Transformers:
 - `attn.py`: attention, Self-Attention, Cross-Attention
 - `embed.py`: embedders
 - `ffn.py`: standard MLP, SwiGLU
 - `norm.py`: modulation, RMSNorm
 - `ropec.py`: RoPE (implemented with complex tensors)
 - `ropem.py`: RoPE (implemented with matrix multiplication)
 - `sinpe.py`: Sinusoidal Positional Embeddings

The code is adapted and re-organized from a wide range of sources, including:
 - GLIDE: https://github.com/openai/glide-text2im
 - MAE: https://github.com/facebookresearch/mae
 - LLaMa: https://github.com/meta-llama/llama
 - DiT: https://github.com/facebookresearch/DiT
 - SiT: https://github.com/willisma/SiT
 - LightningDiT: https://github.com/hustvl/LightningDiT
 - JiT: https://github.com/LTH14/JiT
 - FLUX: https://github.com/black-forest-labs/flux
