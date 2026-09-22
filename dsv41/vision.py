"""Vision encoder and image preprocessing for DeepSeek-V4.1-Flash.

An image is transformed into a `n_vit_h x n_vit_w` patch grid for the ViT and
a `n_llm_h x n_llm_w` token grid after the 3x3 aligner downsampling, which the LLM sees as:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every one of those positions carries `image_token_id` (129264) in `input_ids`; only the
token type distinguishes them. The IMAGE slots are filled with aligner feature vectors
in reading order.
"""

from dataclasses import dataclass
from functools import lru_cache
import base64
import io
import math
import os
import re
from urllib.request import urlopen
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageOps

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_TAG_PATTERN = re.compile(r"<image>(.*?)</image>", re.DOTALL)


@dataclass
class ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


@dataclass
class VisionConfig:
    vision_enabled: bool = True
    vision_n_layers: int = 32
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295936
    vision_max_wh_ratio: float | None = None
    vision_rope_theta: float = 10000.0
    image_token_id: int = 129264
    dim: int = 5120

    @classmethod
    def from_cfg(cls, cfg: dict):
        return cls(
            vision_enabled=bool(cfg.get("vision_n_layers", 0) > 0),
            vision_n_layers=int(cfg.get("vision_n_layers", 32)),
            vision_dim=int(cfg.get("vision_dim", 1024)),
            vision_n_heads=int(cfg.get("vision_n_heads", 16)),
            vision_inter_dim=int(cfg.get("vision_inter_dim", 2816)),
            vision_patch_size=int(cfg.get("vision_patch_size", 14)),
            vision_downsample_ratio=int(cfg.get("vision_downsample_ratio", 3)),
            vision_max_n_token=int(cfg.get("vision_max_n_token", 1024)),
            vision_min_pixels=int(cfg.get("vision_min_pixels", 295936)),
            vision_max_wh_ratio=cfg.get("vision_max_wh_ratio", None),
            vision_rope_theta=float(cfg.get("vision_rope_theta", 10000.0)),
            image_token_id=int(cfg.get("image_token_id", 129264)),
            dim=int(cfg.get("dim", 5120)),
        )


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height: int, width: int, patch_size: int, downsample_ratio: int, max_n_token: int):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height: int, width: int, best_height: int, best_width: int, patch_size: int, downsample_ratio: int, max_n_token: int):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record: dict) -> bytes:
    """Load raw image bytes from base64, URL, local file path, or Anthropic source."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        if os.path.exists(url):
            with open(url, "rb") as file:
                return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def plan_image_grid(width: int, height: int, args: Any):
    p = args.vision_patch_size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)


def load_image(record: dict, args: Any) -> tuple[torch.Tensor, int, int, int, int]:
    """Load and transform an image record into normalized ViT patches [N, 3, p, p]."""
    p = args.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, args)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


def parse_tagged_text(text: str) -> Union[str, List[Dict[str, Any]]]:
    """Convert `<image>path_or_url</image>` occurrences into standard image blocks."""
    matches = list(IMAGE_TAG_PATTERN.finditer(text))
    remaining = IMAGE_TAG_PATTERN.sub("", text)
    if "<image>" in remaining or "</image>" in remaining:
        raise ValueError("Malformed <image>path</image> tag")
    if not matches:
        return text

    blocks: List[Dict[str, Any]] = []
    cursor = 0
    for match in matches:
        if match.start() > cursor:
            blocks.append({"type": "text", "text": text[cursor:match.start()]})
        path = match.group(1).strip()
        if not path:
            raise ValueError("Image path must not be empty")
        blocks.append({
            "type": "image_url",
            "image_url": {"url": path},
        })
        cursor = match.end()
    if cursor < len(text):
        blocks.append({"type": "text", "text": text[cursor:]})
    return blocks


def prepare_vl_inputs(prompt: str, images: list[dict], tokenizer: Any, args: Any):
    """Tokenize prompt, expanding each `<｜deepseek_image｜>` into its 2D token grid span.
    
    Returns (tokens: list[int], token_types: list[int], image_inputs: list[ImageInput]).
    """
    image_token_id = getattr(args, "image_token_id", 129264)
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(f"Found {num_placeholders} image tokens but got {len(images)} images")

    tokens: list[int] = []
    token_types: list[int] = []
    image_inputs: list[ImageInput] = []
    image_iter = iter(images)

    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(next(image_iter), args)
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens += [image_token_id] * types.numel()
        token_types += types.tolist()

    return tokens, token_types, image_inputs or None


# =====================================================================
# ViT & Aligner Architecture
# =====================================================================

@lru_cache(16)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: bidirectional attention over an image patch grid with 2D RoPE."""

    def __init__(self, args: Any):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """Aligns ViT features into LLM token dimension with 3x3 spatial pooling."""

    def __init__(self, args: Any):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class VisionTower(nn.Module):
    """Encapsulates ViT, Aligner, and learned delimiters."""

    def __init__(self, args: Any, device: torch.device | str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.vision = ViT(args).to(device=self.device, dtype=torch.bfloat16)
        self.aligner = Aligner(args).to(device=self.device, dtype=torch.bfloat16)
        self.image_start = nn.Parameter(torch.empty(args.dim, dtype=torch.bfloat16, device=self.device))
        self.image_newline = nn.Parameter(torch.empty(args.dim, dtype=torch.bfloat16, device=self.device))
        self.image_end = nn.Parameter(torch.empty(args.dim, dtype=torch.bfloat16, device=self.device))

    @torch.inference_mode()
    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """Encode image patches into aligned LLM token embeddings [n_llm_h * n_llm_w, dim]."""
        patches = patches.to(device=self.device, dtype=torch.bfloat16)
        vit_feats = self.vision(patches, n_vit_h, n_vit_w)
        return self.aligner(vit_feats, n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images: list[list[ImageInput]] | list[ImageInput], h: torch.Tensor, offset: int = 0):
        """Overwrite image token spans in token embeddings h [B, S, dim] with ViT/aligner features.
        h may be a chunk of the prompt starting at prompt position `offset` (chunked prefill): only the part of each
        image span that falls inside [offset, offset + S) is written, and the ViT features are computed once per image."""
        # Handle both flat list and nested list of samples
        if images and isinstance(images[0], ImageInput):
            sample_list = [images]
        else:
            sample_list = images

        S = h.shape[1]
        for i, sample in enumerate(sample_list):
            for img in sample or ():
                n = img.types.numel()
                lo, hi = max(img.start, offset), min(img.start + n, offset + S)
                if lo >= hi:
                    continue  # this image lies in another chunk
                types_all = img.types.to(h.device)
                types = types_all[lo - img.start : hi - img.start]
                span = h[i, lo - offset : hi - offset]
                span[types == IMAGE_START] = self.image_start.to(device=h.device, dtype=h.dtype)
                span[types == IMAGE_END] = self.image_end.to(device=h.device, dtype=h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(device=h.device, dtype=h.dtype)
                embeds = getattr(img, "_embeds", None)
                if embeds is None:
                    embeds = self.encode_image(img.patches, img.n_vit_h, img.n_vit_w)
                    try:
                        img._embeds = embeds
                    except Exception:
                        pass
                k0 = int((types_all[: lo - img.start] == IMAGE).sum())  # image tokens of this image before the chunk
                k1 = k0 + int((types == IMAGE).sum())
                span[types == IMAGE] = embeds[k0:k1].to(device=h.device, dtype=h.dtype)


def load_vision_tower(ckpt: Any, cfg: dict, dev0: torch.device) -> VisionTower | None:
    """Load ViT, Aligner, and delimiters from checkpoint if vision weights exist."""
    if not cfg.get("vision_n_layers", 0) or "image_start" not in ckpt:
        return None

    env_dev = os.environ.get("DSV41_VISION_DEVICE")
    ALLOWED_VISION_GPUS = (0, 1, 2, 3)
    if env_dev:
        vision_device = torch.device(env_dev)
    else:
        vision_device = dev0
        try:
            for d in ALLOWED_VISION_GPUS:
                dev_obj = torch.device(f"cuda:{d}")
                if dev_obj != dev0:
                    free, _ = torch.cuda.mem_get_info(d)
                    if free > 10 * (1024**3):
                        vision_device = dev_obj
                        break
        except Exception:
            pass

    print(f"loading vision tower onto {vision_device}...", flush=True)
    v_cfg = VisionConfig.from_cfg(cfg)
    tower = VisionTower(v_cfg, device=vision_device)

    # Load ViT weights
    vit_sd = {}
    for name in ckpt.names("vision."):
        param_name = name[len("vision."):]
        vit_sd[param_name] = ckpt.get(name, vision_device).to(torch.bfloat16)
    tower.vision.load_state_dict(vit_sd)
    del vit_sd

    # Load Aligner weights
    aligner_sd = {}
    for name in ckpt.names("aligner."):
        param_name = name[len("aligner."):]
        aligner_sd[param_name] = ckpt.get(name, vision_device).to(torch.bfloat16)
    tower.aligner.load_state_dict(aligner_sd)
    del aligner_sd

    # Load delimiters
    tower.image_start.data.copy_(ckpt.get("image_start", vision_device).to(torch.bfloat16))
    tower.image_newline.data.copy_(ckpt.get("image_newline", vision_device).to(torch.bfloat16))
    tower.image_end.data.copy_(ckpt.get("image_end", vision_device).to(torch.bfloat16))

    print(f"vision tower loaded successfully on {vision_device} (32 ViT layers, Aligner, delimiters)", flush=True)
    return tower
