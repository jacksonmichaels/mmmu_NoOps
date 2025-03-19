import json
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
os.chdir("/home/jacksonmicha_umass_edu/multimodal-modality-conflict")
from dataset.shape_dataset.shape_dataset import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
import pandas as pd
from model.zoo import MODEL_DICT
from transformers import LlavaNextForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoProcessor
from model.llava_code import prepare_inputs_llava, format_prompt_llava, load_model_llava
from torch import nn
from typing import Optional, Tuple
import types
import math



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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    
    color_token = 1560
    shape_token = 1561
    img_mask_ends = (3, 1487)
    
    if self.patch is not None:
        for patch in self.patch:
            head = patch['head']
            val = patch['val']
            if patch['token'] == 'img':
                attn_weights[:, head, :, img_mask_ends[0]:img_mask_ends[1]] += val
            elif patch['token'] == 'txt':
                attn_weights[:, head, :, color_token] += val
    
    
    attn_output = torch.matmul(attn_weights, value_states)
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def apply_patches(layers, patches, coef=0, reset=True, token="img"):
    for idx, layer in enumerate(layers):
        # funcType = type(layer.self_attn.forward)
        layer.self_attn.forward = types.MethodType(custom_forward, layer.self_attn)
        if idx in patches.keys():
            if reset or hasattr(layer.self_attn, "patch") == False:
                layer.self_attn.patch = []
            for head in patches[idx]:
                layer.self_attn.patch.append(
                    {
                        "token": token,
                        "val": torch.tensor(patches[idx][head] * coef).to("cuda"),
                        "head": head
                    }
                )
        else:
            layer.self_attn.patch = None
def eval_model(model, processor, sample, top_k=1):
    with torch.no_grad():
        output = model(**sample, output_attentions=False)
        logits = output.logits[0]
        # idx = torch.argmax(logits[-1])
        idx = torch.topk(logits[-1], top_k).indices
        resp = processor.batch_decode(idx)
        if top_k == 1:
            return resp[0]
        else:
            return resp
        # print(resp)

def get_patches(k):        
    aligned_results = "results/small_attn_aligned/results.json"
    with open(aligned_results, 'r') as f:
        aligned_results = json.load(f)

    results = "results/small_attn_test/results.json"
    with open(results, 'r') as f:
        results = json.load(f)

    for key in results.keys():
        results[key]['aligned_attn_path'] = aligned_results[key]['attn_path']

    filter_colors = ["orange", "brown"]

    targ_img_color = "blue"

    txt_aligned = []
    img_aligned = []
    all_aligned = []
    all_conflict = []
    for key in tqdm(results.keys()):
        sample = results[key]
        if sample['params']["text_color"] in filter_colors or sample['params']["image_color"] in filter_colors:
            continue
        attn_path = sample['attn_path']
        attn_vec = torch.load(attn_path)
        attn_vec[0] *= 1485
        align = sample.get('predicted_answer_alignment', "none")
        if align == 'text':
            txt_aligned.append(attn_vec)
        elif align == 'image':
            img_aligned.append(attn_vec)

        all_conflict.append(attn_vec)

        align_attn_path = sample['aligned_attn_path']
        align_attn_vec = torch.load(attn_path)
        all_aligned.append(align_attn_vec)

    img_aligned = torch.stack(img_aligned)
    txt_aligned = torch.stack(txt_aligned)
    all_aligned = torch.stack(all_aligned)
    all_conflict = torch.stack(all_conflict)
    txt_mean = txt_aligned.mean(dim=0)
    img_mean = img_aligned.mean(dim=0)
    all_align = all_aligned.mean(dim=0)
    all_conf = all_conflict.mean(dim=0)
    
    img_vec =  img_mean - txt_mean

    img_token_patch = torch.topk(img_vec[0].flatten(), k)
    txt_token_patch = torch.topk(-img_vec[1].flatten(), k)
    return img_token_patch, txt_token_patch

    
def main(k):
    output_path = f"top{k}_results.json"
    device = "cuda"
    dset = ShapeDataset("dataset/shape_dataset/conflict_set_metadata.json")
    model_path = MODEL_DICT["llava-one-7b"]['model_path']
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    call_engine_fn = prepare_inputs_llava
    prompt_format_fn = format_prompt_llava

    img_token_patch, txt_token_patch = get_patches(1)


    targ_idxs = None
    targ_idxs = [
        655, 
        593, 
        629
    ]
    
    targ_idxs = [targ_idxs[k]]
    
    img_align_patch = {
    }

    for val, idx in zip(img_token_patch.values, img_token_patch.indices):
        if targ_idxs is not None:
            if idx not in targ_idxs:
                continue
        layer = idx // 28
        head = idx % 28

        if layer not in img_align_patch.keys():
            img_align_patch[int(layer)] = {}
        img_align_patch[int(layer)][int(head)] = float(val)

    txt_align_patch = {
    }

    for val, idx in zip(txt_token_patch.values, txt_token_patch.indices):
        if targ_idxs is not None:
            if idx not in targ_idxs:
                continue
        layer = idx // 28
        head = idx % 28

        if layer not in txt_align_patch.keys():
            txt_align_patch[int(layer)] = {}
        txt_align_patch[int(layer)][int(head)] = float(val)


    layers = model.language_model.model.layers

    color_token = 1560
    shape_token = 1561
    img_mask_ends = (3, 1487)

    coefs = torch.arange(-0.2, 0.2, 0.02)
    
    img_align = []
    txt_align = []
    no_align = []

    for coef in tqdm(coefs, leave=False):
        img_align.append(0)
        txt_align.append(0)
        no_align.append(0)
        apply_patches(layers, img_align_patch, -1 * coef)
        apply_patches(layers, txt_align_patch, coef, False, "txt")

        with torch.no_grad():
            for idx in tqdm(range(900), leave=False):
                sample = dset[idx + 4500]
                new_sample = prompt_format_fn(sample, text_input_type="caption", conflict_type="direct_alternative")
                final_sample = call_engine_fn(new_sample, processor, device)
                resp = eval_model(model, processor, final_sample, 1)

                if resp == sample['params']['image_color']:
                    img_align[-1] += 1
                elif resp == sample['params']["text_color"]:
                    txt_align[-1] += 1
                else:
                    no_align[-1] += 1
    ret_list = {
        "img_align": img_align,
        "txt_align": txt_align,
        "no_align": no_align,
    }
    
    out_path = f"{k}_results_single_head.json"
    with open(out_path, 'w') as f:
        json.dump(ret_list, f)

if __name__ == "__main__":
    for k in tqdm(range(3)):
        main(k)