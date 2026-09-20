import os
import sys
import torch
from dsv41.vision import (
    VisionConfig,
    VisionTower,
    load_image,
    prepare_vl_inputs,
    parse_tagged_text,
    IMAGE_START,
    IMAGE,
    IMAGE_NEW_LINE,
    IMAGE_END,
    IMAGE_PLACEHOLDER,
)
from transformers import AutoTokenizer

def test_tagged_text():
    raw = "Here is an image: <image>/path/to/test.jpg</image> What is this?"
    blocks = parse_tagged_text(raw)
    assert isinstance(blocks, list)
    assert len(blocks) == 3
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"] == "/path/to/test.jpg"
    assert blocks[2]["type"] == "text"
    print("✓ test_tagged_text passed")

def test_prepare_vl_inputs():
    ckpt_path = "/mnt/ssd/models/DeepSeek-V4.1-Flash"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    cfg = {
        "vision_n_layers": 32,
        "vision_dim": 1024,
        "vision_n_heads": 16,
        "vision_inter_dim": 2816,
        "vision_patch_size": 14,
        "vision_downsample_ratio": 3,
        "vision_max_n_token": 1024,
        "vision_min_pixels": 295936,
        "vision_rope_theta": 10000.0,
        "image_token_id": 129264,
        "dim": 5120
    }
    v_cfg = VisionConfig.from_cfg(cfg)

    prompt = f"<｜begin▁of▁sentence｜><｜User｜>Describe this: {IMAGE_PLACEHOLDER}<｜Assistant｜></think>"
    images = [{"url": "/mnt/ssd/models/DeepSeek-V4.1-Flash/inference/examples/images/carrots.jpeg"}]

    tokens, token_types, image_inputs = prepare_vl_inputs(prompt, images, tokenizer, v_cfg)
    assert len(tokens) == len(token_types)
    assert len(image_inputs) == 1
    img = image_inputs[0]
    assert img.types.numel() == 444
    assert img.types[0].item() == IMAGE_START
    assert img.types[-1].item() == IMAGE_END
    print("✓ test_prepare_vl_inputs passed")

def test_merge_image_embeddings():
    cfg = {
        "vision_n_layers": 32,
        "vision_dim": 1024,
        "vision_n_heads": 16,
        "vision_inter_dim": 2816,
        "vision_patch_size": 14,
        "vision_downsample_ratio": 3,
        "vision_max_n_token": 1024,
        "vision_min_pixels": 295936,
        "vision_rope_theta": 10000.0,
        "image_token_id": 129264,
        "dim": 5120
    }
    v_cfg = VisionConfig.from_cfg(cfg)
    device = torch.device("cuda:4" if torch.cuda.is_available() and torch.cuda.device_count() > 4 else "cpu")
    tower = VisionTower(v_cfg, device=device)
    tower.image_start.data.fill_(1.0)
    tower.image_newline.data.fill_(2.0)
    tower.image_end.data.fill_(3.0)

    img_record = {"url": "/mnt/ssd/models/DeepSeek-V4.1-Flash/inference/examples/images/carrots.jpeg"}
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(img_record, v_cfg)
    from dsv41.vision import image_token_types, ImageInput
    types = image_token_types(n_llm_h, n_llm_w)
    img_input = ImageInput(start=5, patches=patches, n_vit_h=n_vit_h, n_vit_w=n_vit_w, types=types)

    B = 1
    S = 5 + types.numel() + 10
    h = torch.zeros((B, S, 5120), dtype=torch.bfloat16, device="cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")

    tower.merge_image_embeddings([img_input], h)
    # Check that delimiters were placed
    assert not torch.all(h[0, 5] == 0)
    assert not torch.all(h[0, 5 + types.numel() - 1] == 0)
    print("✓ test_merge_image_embeddings passed")

if __name__ == "__main__":
    test_tagged_text()
    test_prepare_vl_inputs()
    test_merge_image_embeddings()
    print("All vision unit tests passed successfully!")
