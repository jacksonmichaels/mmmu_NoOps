#!/usr/bin/env python3

import sys
import os
import json
import torch
import random
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
import re
import ast
from pathlib import Path
from typing import Optional, Tuple
from types import MethodType
from torchvision.transforms.functional import pil_to_tensor
# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from transformers import LlavaNextForConditionalGeneration  # for "LLAVA" model_type
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import AutoModelForCausalLM
from torchvision.transforms.functional import InterpolationMode

from utils.model_utils import build_transform, find_closest_aspect_ratio, dynamic_preprocess, load_image

# --------------------------------------------------------------------------------
# 1) MODEL DICTIONARY (INCLUDING LLaVA-Next Interleave Qwen-7B)
# --------------------------------------------------------------------------------
CACHE_DIR_BASE = "/u/li19/data_folder/model_cache"

MODEL_DICT_LLMs = {
    "idefics": {
        "model_id": "HuggingFaceM4/idefics2-8b-base",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "AutoModelForVision2Seq",
    },
    "llava_next": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "LLAVA",
    },
    "llava_one": {
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "llava_one",
    },
    "llava_one_0.5b": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "llava_one",
    },
    # -------------------------------------------------------------------------
    # NEW ENTRY FOR InternVL 2.5
    # -------------------------------------------------------------------------
    "internvl2_5": {
        "model_id": "OpenGVLab/InternVL2_5-8B",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "internvl2_5",
    },
    # Additional existing or custom models can go here...
}

# --------------------------------------------------------------------------------
# 2) LOAD MMMU DATASET
# --------------------------------------------------------------------------------
CAT_SHORT2LONG = {
    "math": "Mathematics",
    "science": "Science",
    "history": "History",
    "chemistry": "Chemistry",
    "biology": "Biology",
    # etc.
}

all_subs = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory','Basic_Medical_Science','Biology','Chemistry','Clinical_Medicine',
    'Computer_Science','Design','Diagnostics_and_Laboratory_Medicine','Economics',
    'Electronics','Energy_and_Power','Finance','Geography','History','Literature',
    'Manage','Marketing','Materials','Math','Mechanical_Engineering','Music',
    'Pharmacy','Physics','Psychology','Public_Health','Sociology'
]

def load_dataset(data_path, subject, split="validation"):
    from datasets import load_dataset
    ds = load_dataset(data_path, subject, split=split)
    return ds

def load_noop(data_path, subject, split="validation", noop_root="/u/li19/MMMU/mmmu-noop", noOps='text'):
    from datasets import load_dataset
    # Load your dataset
    ds = load_dataset(data_path, subject, split=split)
    
    if noOps == "text" or noOps == "all":
        # Load the NoOp data from JSON
        noop_file = os.path.join(noop_root, f"mmmu_{split}_noop_insert_sentence.json")
        with open(noop_file, 'r') as f:
            noop_data = json.load(f)
        
    if noOps == "img" or noOps == "all":
        # Load the NoOp data from JSON
        img_noop_file = os.path.join(noop_root, f"val_img_NoOp_metadata.json")
        with open(img_noop_file, 'r') as f:
            img_data = json.load(f)
    
    # Define a function that adds "noop" to each example
    def add_noop(example, noOps):
        example_id = example['id']
        if noOps == "text" or noOps == "all":
            # print("adding text noop")
            example['noop'] = noop_data.get(example_id, None)  # fallback if ID not found
        if noOps == "img" or noOps == "all":
            example['noop_imgs'] = True
            img_datum = img_data[example_id]['image_paths']
            image_names = [f"image_{n}" for n in range(1, 8)]
            imgs = [(name, example[name]) for name in image_names if example[name] is not None]
            for img_pair, imgNoOp in zip(imgs, img_datum):
                name, img = img_pair
                example[name] = Image.open(imgNoOp + ".png")
                # print("adding img noop: ", name)
        return example

    # Use .map() to apply the function and return a new dataset with the `noop` column
    ds = ds.map(lambda x: add_noop(x, noOps))

    return ds

def concatenate_datasets(sub_dataset_list):
    from datasets import concatenate_datasets
    combined = concatenate_datasets(sub_dataset_list)
    return combined

# --------------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def save_json(path, data):
    path = Path(path)
    parent = path.parent.absolute()
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_yaml(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_single_sample(sample):
    if "image_path" in sample and os.path.isfile(sample["image_path"]):
        img = Image.open(sample["image_path"]).convert("RGB")
        sample["image"] = img
    return sample

def construct_prompt(sample, config):
    question = sample.get("question", "")
    sample["final_input_prompt"] = f"Question: {question}\nAnswer:"
    return sample

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--config_path", type=str, default="configs/llava1.5.yaml")
    parser.add_argument("--data_path", type=str, default="MMMU/MMMU",
                        help="Hugging Face dataset path for MMMU")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Key in MODEL_DICT_LLMs (e.g. 'llava_one' or 'internvl2_5')")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str, default=None,
                        help="If model is private, specify your HF token here.")
    parser.add_argument("--noop", type=str, default="none",  help="Define what NoOps to add [text, img, all, none]")
    parser.add_argument("--subject", type=str, default="all")
    parser.add_argument('-w', '--warn_noop',action='store_true')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--store_attention",action='store_true')

    return parser.parse_args()

def set_device_and_seed(seed, device):
    torch.manual_seed(seed)
    if device == "cuda" or device == "cpu":
        return device
    else:
        return "cuda:"+device

def load_config(config_path):
    print(f"Loading config from {config_path} ...")
    config = load_yaml(config_path)
    for k, v in list(config.items()):
        if k != "eval_params" and isinstance(v, list) and len(v) == 1:
            config[k] = v[0]
    return config

def prepare_NoOp(data_path, split, all_subs, noOps='text'):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_noop(data_path, subject, split=split, noOps=noOps) for subject in all_subs]
    dataset = concatenate_datasets(sub_dataset_list)
    print("NoOp Dataset loaded:", dataset)
    return dataset

def prepare_dataset(data_path, split, all_subs):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_dataset(data_path, subject, split=split) for subject in all_subs]
    dataset = concatenate_datasets(sub_dataset_list)
    print("Dataset loaded:", dataset)
    return dataset

def handle_model_types(model_type, model_id, cache_dir, args, device):
    """
    Fallback logic or specialized load logic for certain model types.
    """
    if model_type == "AutoModelForCausalLM":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        ).to(device)
        processor = None

    else:
        # Fallback
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        ).to(device)
        processor = None

    return tokenizer, processor, model

def load_model_and_tokenizer(args, device):
    if args.model_name not in MODEL_DICT_LLMs.keys():
        raise ValueError(f"Model '{args.model_name}' not found in MODEL_DICT_LLMs.")

    model_info = MODEL_DICT_LLMs[args.model_name]
    model_id = model_info["model_id"]
    cache_dir = model_info["cache_dir"]
    model_type = model_info.get("model_type", "AutoModel")

    print(f"Loading model '{args.model_name}' from '{model_id}' ...")

    if model_type == "LLAVA":
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            use_auth_token=args.access_token,
            trust_remote_code=True
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            use_auth_token=args.access_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_auth_token=args.access_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

    elif model_type == "llava_one":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            use_auth_token=args.access_token,
            trust_remote_code=True,
        ).to(device)
        tokenizer = None
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_auth_token=args.access_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        if args.store_attention:
            # Optionally, if you want to keep a reference to the original forward method:
            for layer in model.language_model.model.layers:
                obj = layer.self_attn
                obj._original_forward = obj.forward  # save the original forward if needed
                obj.forward = MethodType(verbose_forward, obj)

    # -------------------------------------------------------------------------
    # NEW LOADING LOGIC FOR InternVL2_5
    # -------------------------------------------------------------------------
    elif model_type == "internvl2_5":
        # As per HF docs, we can load it with AutoModel, and use tokenizer = AutoTokenizer
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # You can switch to bfloat16 if desired
            cache_dir=cache_dir,
            use_auth_token=args.access_token,
            trust_remote_code=True
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_auth_token=args.access_token,
            cache_dir=cache_dir
        )
        # InternVL2.5 uses the model's `.chat()` method
        processor = None

    else:
        # Fallback or handle any other custom model types
        tokenizer, processor, model = handle_model_types(model_type, model_id, cache_dir, args, device)

    print("Model loaded successfully!")
    return model, tokenizer, processor
    
# --------------------------------------------------------------------------------
# INTERN VLM FUNCTIONS
# --------------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # A small heuristic to break ties, if needed
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Splits an image into tiles based on a dynamic aspect ratio approach.
    If use_thumbnail=True, we also add a thumbnail at the end if >1 tiles.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate candidate (rows,cols) up to max_num
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if (i * j) <= max_num and (i * j) >= min_num
    )
    # Sort by how many tiles, so we try smaller tile grids first
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Pick the best ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    # Calculate new size
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize once, then crop out each tile
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        # each tile's bounding box
        col_size = target_width // image_size
        row_size = target_height // image_size
        box = (
            (i % col_size) * image_size,
            (i // col_size) * image_size,
            ((i % col_size) + 1) * image_size,
            ((i // col_size) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks, \
        f"Expected {blocks} tiles, got {len(processed_images)}"

    # Optionally add a thumbnail as the last patch if there's >1 tile
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def preprocess_internvl2_5_images(pil_images, max_num=12, image_size=448):
    """
    Given a list of PIL images, returns a single stacked tensor of shape:
      (total_tiles, 3, image_size, image_size)
    If no images provided, returns None.
    """
    transform = build_transform(input_size=image_size)
    tile_tensors = []
    for img in pil_images:
        # 1) do dynamic tiling
        tiles = dynamic_preprocess(img, image_size=image_size,
                                   use_thumbnail=True, max_num=max_num)
        # 2) transform each tile -> (3, H, W)
        for tile in tiles:
            tile_tensors.append(transform(tile))

    if not tile_tensors:
        return None  # no images

    # concat them all
    pixel_values = torch.stack(tile_tensors, dim=0)  # shape: (N,3,H,W)
    return pixel_values

# --------------------------------------------------------------------------------
# PROMPT CONSTRUCTION & PARSING HELPERS
# --------------------------------------------------------------------------------
def string_to_dict(string):
    content = []
    image_placeholder = {"type": "image"}
    image_tokens = []
    pattern = r"<image (?:10|[0-9])>"
    matches = re.findall(pattern, string)
    image_tokens.extend(matches)
    split_question = re.split(pattern, string)
    for idx, segment in enumerate(split_question):
        if segment.strip():
            content.append({"type": "text", "text": segment})
        if idx < len(split_question) - 1:
            content.append(image_placeholder)
    return content, image_tokens

def construct_prompt_l1(sample):
    if "noop" in sample.keys():
        question = sample.get("noop", "")
    else:
        question = sample.get("question", "")

        
    image_names = [f"image_{idx}" for idx in range(1,8)]
    images = []
    for name in image_names:
        if name in sample and sample[name] is not None:
            images.append(sample[name])
    sample["images"] = []
    conversation = []
    image_placeholder = {"type": "image"}

    pattern = r"<image (?:10|[0-9])>"
    split_question = re.split(pattern, question)

    if sample['question_type'] == "multiple-choice":
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For multiple choice questions, please only reply with a single letter for your answer choice.")
        # instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately. "
        #                "please fully explain your reasoning steps behind your answer and end your reply with your final choice in brackets.")
    else:
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For open questions, reply as succinctly as possible with the final answer only.")

    if sample['noop_warning']:
        warn_text = "This might be a trick question designed to confuse LLMs with additional information. Look for irrelevant information or distractors in the question:"
        instruction += "\n" + warn_text

    content = [{"type": "text", "text": instruction}]
    question_dict, question_images = string_to_dict(question)
    content += question_dict
    sample["images"] += question_images

    if sample['question_type'] == "multiple-choice":
        answers = ast.literal_eval(sample['options'])
        ops = 'ABCDEFGHIJKL'
        content.append({"type": "text", "text": "Your options are: "})
        for i, ans in enumerate(answers):
            content.append({"type": "text", "text": ops[i] + ") "})
            answer_dict, answer_imgs = string_to_dict(ans)
            sample["images"] += answer_imgs
            content += answer_dict

    response = {
        "role": "user",
        "content": content
    }
    sample["final_input_prompt"] = [response]
    return sample


def parse_llava_1v(response):
    text = response[0]
    ans = text.split("\n")[-1].replace(")", "")
    return ans

def parse_multi_choice_response(response, all_choices, index2ans):
    resp_lower = response.lower()
    for choice in all_choices:
        if choice.lower() in resp_lower:
            return choice
    return response

def hf_image_processor(image):
    return image

# --------------------------------------------------------------------------------
# 4) THE INFERENCE ENGINE
# --------------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def verbose_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

def verbose_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output, attn_weight = verbose_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weight

def verbose_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attn_output, attn_weights = verbose_sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

def call_llava_one_engine(args, sample, model, processor):
    """
    Inference using the LlavaOnevisionForConditionalGeneration model.
    """
    device = next(model.parameters()).device
    prompt_text = sample.get("final_input_prompt")
    prompt = processor.apply_chat_template(prompt_text, add_generation_prompt=True)

    # Convert 'images' from strings to PIL if needed
    for i in range(len(sample['images'])):
        if isinstance(sample['images'][i], str):
            sample['images'][i] = sample[
                sample['images'][i].replace(" ", "_").replace("<", "").replace(">","")
            ]

    inputs = processor(
        text=[prompt],
        images=sample['images'],
        return_tensors="pt",
        padding=True
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)

        
    generation_config = {
        "max_new_tokens": 256,              # Limit the number of tokens generated
        "temperature": 1.0,               # Neutral temperature; does not affect greedy decoding
        "do_sample": False,               # Disable sampling
        "num_beams": 1,                   # Greedy decoding (no beam search)
        "repetition_penalty": 1.0,        # No penalty for repeated tokens
        "length_penalty": 1.0,            # No length bias
        "early_stopping": True,           # Stop as soon as EOS is reached
        "pad_token_id":processor.tokenizer.eos_token_id
    }


    # for ids in inputs.input_ids[0].cpu():
    #     print()
    #     print(ids)
    #     print(processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    IMAGE_TOKEN = 151646
    img_mask = (inputs.input_ids == IMAGE_TOKEN)[0]
    txt_mask = ~img_mask  # This creates the complementary mask on the same device

    # txt_mask = img_mask

    if args.store_attention:
        with torch.no_grad():
            response = model(
                **inputs, output_attentions=True
            )
            # print(response.keys())
            logits = response.logits.detach().cpu()
            attn = torch.stack([el for el in response.attentions]).detach()
            img_attn = attn[:,:,:,img_mask,:][:,:,:,:,img_mask]
            txt_attn = attn[:,:,:,txt_mask,:][:,:,:,:,txt_mask]
            attn_tuple = [img_attn.mean(dim=[1,3,4]).detach().cpu(), txt_attn.mean(dim=[1,3,4]).detach().cpu()]
            predicted_token_id = logits[0, -1].argmax().item()
            predicted_token_str = processor.decode(predicted_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            del img_mask, txt_mask, response, logits, inputs, attn
            torch.cuda.empty_cache()
        return predicted_token_str, attn_tuple

    else:
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, **generation_config
            )
            responses = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        return responses, None



def replace_image_tokens(text):
    pattern = re.compile(r'<image\s+(\d+)>')
    images_replaced = []
    
    # This will be our counter for "x"
    replacement_count = 0
    
    def _replacer(match):
        nonlocal replacement_count
        
        # Extract the original image number
        original_number = match.group(1)
        
        # Increment counter to create "Image-x<image>"
        replacement_count += 1
        new_token = f"Image-{replacement_count}<image>"
        
        # Record the replacement
        images_replaced.append({
            "image_new": new_token,
            "image_old": original_number
        })
        
        return new_token
    
    # Perform the replacement
    modified_text = pattern.sub(_replacer, text)
    
    return modified_text, images_replaced

def construct_prompt_intern(sample):
    if "noop" in sample.keys():
        question = sample.get("noop", "")
    else:
        question = sample.get("question", "")
        
        
    image_names = [f"image_{idx}" for idx in range(1,8)]
    images = []

    if sample['question_type'] == "multiple-choice":
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For multiple choice questions, please only reply with a single letter for your answer choice.")
    else:
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For open questions, reply as succinctly as possible with the final answer only.")

    op_text = ""
    if sample['question_type'] == "multiple-choice":
        answers = ast.literal_eval(sample['options'])
        ops = 'ABCDEFGHIJKL'
        op_text += "Your options are:\n"
        for i, ans in enumerate(answers):
            op_text += ops[i] + ") " + ans + "\n"


    response = instruction + "\n" + question + "\n" + op_text
    pattern = r"<image (?:10|[0-9])>"

    response, images = replace_image_tokens(response)
    pixel_values = []
    img_sizes = []
    for img in images:
        pixel_values.append(
            load_image(sample[f"image_{img['image_old']}"])
        )
        img_sizes.append(pixel_values[-1].size(0))
    
    sample["final_input_prompt"] = [response]
    sample["pixel_values"] = torch.cat(pixel_values, dim=0)
    sample["image_sizes"] = torch.tensor(img_sizes)
    return sample
    
def call_internvl2_5_engine(args, sample, model, tokenizer):
    """
    Inference using the InternVL2.5 model from OpenGVLab (which uses .chat()).
    Now with full image transformation logic for any real PIL images found in sample["images"].
    """

    if "noop" in sample.keys():
        question = sample.get("noop", "")
    else:
        question = sample.get("question", "")
        
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # 4) Use the .chat() method for inference
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        response = model.chat(
            tokenizer, 
            sample['pixel_values'].to(device = device, dtype = dtype), 
            question, 
            generation_config,
            num_patches_list=sample["image_sizes"].to(device = device, dtype = int), 
            return_history=False
        )

    return [response]
# --------------------------------------------------------------------------------
# 5) RUN INFERENCE
# --------------------------------------------------------------------------------
def run_inference(dataset, model, processor, args, tokenizer=None):
    model_type = MODEL_DICT_LLMs[args.model_name].get("model_type", "llava_one")
    samples = []
    print("Processing Dataset")
    for sample in tqdm(dataset):
        sample['noop_warning'] = args.warn_noop
        if model_type == "llava_one":
            sample = process_single_sample(sample)
            sample = construct_prompt_l1(sample)
            samples.append(sample)
        elif model_type == "internvl2_5":
            # Pass your image-loading function here (replace lambda as needed)
            sample = construct_prompt_intern(sample, load_image_func=lambda x: x)
            samples.append(sample)
    out_samples_eval_only = {}
    out_samples_parse_and_eval = {}
    model.eval()

    print("Running inference on MMMU dataset...")
    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            sample_id = sample.get("id", str(hash(sample.get("question", ""))))
            subject = "_".join(sample.get("id").split("_")[1:-1]) if "id" in sample else "unknown"
            if model_type == "llava_one":
                if args.store_attention:
                    # Get both the predicted answer and attention data
                    pred_ans, attn = call_llava_one_engine(args, sample, model, processor)
                    # Create a directory to store attention if it doesn't exist
                    attention_dir = os.path.join("results", "attention")
                    os.makedirs(attention_dir, exist_ok=True)
                    # Define a unique file name based on sample_id
                    attn_path = os.path.join(attention_dir, f"{sample_id}_attn.pt")
                    # Save the attention tensor to disk
                    torch.save(attn, attn_path)
                else:
                    response, _ = call_llava_one_engine(args, sample, model, processor)
                    pred_ans = parse_llava_1v(response)
                    attn = None
                    
            elif model_type == "internvl2_5":
                response = call_internvl2_5_engine(args, sample, model, tokenizer)
                pred_ans = response[0]
                return 0  # early return for demonstration
            else:
                response = ["[NOT IMPLEMENTED YET]"]
                pred_ans = response[0]

            
            out_samples_eval_only[sample_id] = pred_ans
            
            parsed_sample = {
                "id": sample_id,
                "question_type": sample.get("question_type", ""),
                "answer": sample.get("answer", ""), 
                "response": pred_ans,
                "attn_path": attn
            }
            if sample.get("question_type") == "multiple-choice":
                parsed_sample["all_choices"] = ["A", "B", "C", "D"]
                parsed_sample["index2ans"] = sample.get("index2ans", {})
            if subject not in out_samples_parse_and_eval:
                out_samples_parse_and_eval[subject] = []
            out_samples_parse_and_eval[subject].append(parsed_sample)
            if i % 250 == 0:
                print(sample)
                print("PRED: ", pred_ans, " REAL: ", sample.get("answer", ""))
        except Exception as e:
            print("Failed on sample: \n ============== \n", sample)
            import traceback
            print(traceback.format_exc())
            print(e)
            break
    return out_samples_eval_only, out_samples_parse_and_eval

def save_results(output_path, eval_only_results, parse_and_eval_results):
    eval_only_path = os.path.join(output_path, "eval_only_output.json")
    save_json(eval_only_path, eval_only_results)
    print(f"Evaluation-only results saved to {eval_only_path}")

    parse_and_eval_path = os.path.join(output_path, "parse_and_eval")
    os.makedirs(parse_and_eval_path, exist_ok=True)
    for subject, subject_data in parse_and_eval_results.items():
        subject_path = os.path.join(parse_and_eval_path, subject)
        os.makedirs(subject_path, exist_ok=True)
        output_file = os.path.join(subject_path, "output.json")
        save_json(output_file, subject_data)
    print(f"Parse-and-evaluation results saved to {parse_and_eval_path}")

# --------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------
def main():
    args = parse_arguments()
    if args.subject == "all":
        subs = all_subs
    else:
        subs = [args.subject]
    device = set_device_and_seed(args.seed, args.device)
    args.config = load_config(args.config_path)

    if args.noop != "none":
        dataset = prepare_NoOp(args.data_path, args.split, subs, args.noop)
    else:
        dataset = prepare_dataset(args.data_path, args.split, subs)

    model, tokenizer, processor = load_model_and_tokenizer(args, device)
    eval_only_results, parse_and_eval_results = run_inference(dataset, model, processor, args, tokenizer)
    
    save_results("results/" + args.output_path, eval_only_results, parse_and_eval_results)

if __name__ == "__main__":
    main()
