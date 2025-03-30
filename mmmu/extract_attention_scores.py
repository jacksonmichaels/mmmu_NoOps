"""
This script is used to extract hidden states from the model.
"""
#%%
import torch
import os
import random
import re
import ast
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import importlib
import json
from datasets import concatenate_datasets, load_dataset
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

# Set grad enabled to False
torch.set_grad_enabled(False)
from functools import partial
import pickle
#%%
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

def prepare_NoOp(data_path, split, all_subs, noOps='text'):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_noop(data_path, subject, split=split, noOps=noOps) for subject in tqdm(all_subs)]
    dataset = concatenate_datasets(sub_dataset_list)
    print("NoOp Dataset loaded:", dataset)
    return dataset

def prepare_dataset(data_path, split, all_subs):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_dataset(data_path, subject, split=split) for subject in tqdm(all_subs)]
    dataset = concatenate_datasets(sub_dataset_list)
    print("Dataset loaded:", dataset)
    return dataset

def load_dataset(noop='all'):
    all_subs = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory','Basic_Medical_Science','Biology','Chemistry','Clinical_Medicine',
    'Computer_Science','Design','Diagnostics_and_Laboratory_Medicine','Economics',
    'Electronics','Energy_and_Power','Finance','Geography','History','Literature',
    'Manage','Marketing','Materials','Math','Mechanical_Engineering','Music',
    'Pharmacy','Physics','Psychology','Public_Health','Sociology'
    ]
    all_subs = ['Accounting', 'Art']
    data_path = "MMMU/MMMU"
    split = "validation"
    subs = all_subs
    if noop != "none":
        dataset = prepare_NoOp(data_path, split, subs, noop)
    else:
        dataset = prepare_dataset(data_path, split, subs)
    return dataset

def load_model(model_name, device="cuda:0"):
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    return processor, model

# def load_model(model_name, device = "cuda"):
#     model = processor = call_engine_fn = prompt_format_fn = None
#     model_path = MODEL_DICT[model_name]['model_path']
#     if model_name.startswith("llava-one"):
#         from model.llava_code import prepare_inputs_llava, format_prompt_llava, load_model_llava
#         model, processor = load_model_llava(model_path, device)
#         call_engine_fn = prepare_inputs_llava
#         prompt_format_fn = format_prompt_llava
        
#     if model_name == "idefics":
#         from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
#         from model.idefics_code import call_idefics_engine, format_prompt_for_idefics
        
#         call_engine_fn = call_idefics_engine
#         prompt_format_fn = format_prompt_for_idefics
#         processor = Idefics2Processor.from_pretrained(model_path)
#         model = Idefics2ForConditionalGeneration.from_pretrained(model_path)
#         model.to(device)
    
#     if model_name == "blip-2":
#         from model.blip_code import load_model_blip, call_engine_blip, format_prompt_blip
#         model, processor = load_model_blip(model_path, device)
#         call_engine_fn = call_engine_blip
#         prompt_format_fn = format_prompt_blip
        
        
#     nnsight_model = NNsight(model)
#     call_engine_fn = partial(call_engine_fn, processor=processor, device=device)
#     return nnsight_model, processor, call_engine_fn, prompt_format_fn


def find_target_segments(x, target=151646):
    """
    Returns a tensor of shape [n, 2], where each row [start, end]
    indicates that x[start:end] is a contiguous block with value `target`.
    """
    # Create a boolean mask where elements equal target.
    mask = (x == target)
    
    # If no elements match the target, return an empty tensor.
    if not mask.any():
        return torch.empty((0, 2), dtype=torch.int64, device=x.device)
    
    # Convert the boolean mask to integers (0 or 1) to compute differences.
    mask_int = mask.to(torch.int)
    # Compute differences between consecutive elements.
    diff = mask_int[1:] - mask_int[:-1]
    
    # A diff value of 1 indicates a transition from False to True (start of a segment)
    # A diff value of -1 indicates a transition from True to False (end of a segment)
    start_indices = torch.nonzero(diff == 1).flatten() + 1
    end_indices = torch.nonzero(diff == -1).flatten() + 1
    
    # Edge case: if the first element is True, then a segment starts at index 0.
    if mask[0]:
        start_indices = torch.cat((torch.tensor([0], device=x.device), start_indices))
    
    # Edge case: if the last element is True, then a segment continues to the end.
    if mask[-1]:
        end_indices = torch.cat((end_indices, torch.tensor([len(x)], device=x.device)))
    
    # Stack start and end indices into a [n,2] tensor.
    segments = torch.stack((start_indices, end_indices), dim=1)
    return segments

def find_inserted_section_indices(A, B):
    # A and B are 1D tensors.
    lenA = A.size(0)
    lenB = B.size(0)
    min_len = min(lenA, lenB)
    
    # Compute prefix equality for the first min_len elements.
    prefix_eq = (A[:min_len] == B[:min_len])
    # Find first index where they differ.
    if prefix_eq.all():
        prefix_match_length = min_len
    else:
        # nonzero returns a 2D tensor; we take the first index.
        prefix_match_length = torch.nonzero(~prefix_eq, as_tuple=False)[0].item()
    
    # Compute suffix equality by reversing the tensors.
    revA = A.flip(0)
    revB = B.flip(0)
    min_len_rev = min(lenA, lenB)
    suffix_eq = (revA[:min_len_rev] == revB[:min_len_rev])
    if suffix_eq.all():
        suffix_match_length = min_len_rev
    else:
        suffix_match_length = torch.nonzero(~suffix_eq, as_tuple=False)[0].item()
    
    # The inserted section in B is located between the matching prefix and the matching suffix.
    start_index = prefix_match_length
    end_index = lenB - suffix_match_length
    return torch.tensor([start_index, end_index])

def sample_to_model_inputs(sample, question, processor):
    for i in range(len(sample['images'])):
        if isinstance(sample['images'][i], str):
            sample['images'][i] = sample[
                sample['images'][i].replace(" ", "_").replace("<", "").replace(">","")
            ]
    # response = [{"role": "user", "content": question}]
    response = question

    prompt = processor.apply_chat_template(response, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        images=sample['images'],
        return_tensors="pt",
        padding=True
    )
    return inputs

def final_selection_mask(B, inserted_indices, target=151646):
    """
    Given a tensor B, inserted_indices (a tuple (start, end) of the inserted section),
    and a target integer value, return a boolean mask selecting tokens that are neither:
      - equal to the target, nor
      - in the inserted section.
      
    Parameters:
      B (torch.Tensor): 1D tensor of tokens.
      inserted_indices (tuple): A tuple (start, end) indicating the inserted segment indices.
      target (int): The target value to exclude.
      
    Returns:
      torch.BoolTensor: A boolean tensor of the same length as B.
    """
    ins_start, ins_end = inserted_indices
    # Create an index tensor [0, 1, ..., len(B)-1]
    indices = torch.arange(B.size(0), device=B.device)
    
    # Build the mask:
    #   - B != target ensures tokens are not equal to the target.
    #   - (indices < ins_start) | (indices >= ins_end) ensures the token is not in the inserted section.
    mask = (B != target) & ((indices < ins_start) | (indices >= ins_end))
    return mask

def extract_attention(dset, model, processor, out_dir,store_all=False,layers=None,device="cuda:0"):
    out_samples = {}
    with torch.no_grad():
        for sample in tqdm(dset):

            base = sample_to_model_inputs(sample, sample['final_input_prompt_base'], processor).to(device)
            noop = sample_to_model_inputs(sample, sample['final_input_prompt_noop'], processor).to(device)
            noop_indices = find_inserted_section_indices(base.input_ids[0], noop.input_ids[0])
            img_segments = find_target_segments(noop.input_ids[0])

            
            output = model(**noop,output_attentions=True)
            logits = output.logits[0, -1] # get the logit of the last token
            top_logit = torch.argmax(logits)
            attn = torch.stack(output.attentions).detach().cpu()
            
            attn_path_folder = os.path.join(
                # "/scratch/workspace/jacksonmicha_umass_edu-attention_scores/attention_scores",
                out_dir, "attention_scores"
            )

            os.makedirs(attn_path_folder, exist_ok=True)
            attn_path = os.path.join(attn_path_folder, sample['id'] + ".pt")
            # async_save(attn, attn_path)

            if not store_all:
                color_idx, shape_idx = (1560, 1561)
                shape_range = (3, 1488)
                img_attn = attn[:, 0, :, -1, img_segments[0]:img_segments[1]].mean(dim=2)
                noop_attn = attn[:, 0, :, -1, noop_indices[0]:noop_indices[1]].mean(dim=2)

                not_noop_mask = final_selection_mask(noop.input_ids[0], noop_indices)
                
                txt_attn = attn[:, 0, :, -1, noop_indices].mean(dim=2)
                stacked = torch.stack([img_attn, noop_attn, txt_attn])
                torch.save(stacked, attn_path)
            else:
                if layers is not None:
                    torch.save(attn[:,0,layers,:,:], attn_path)
                else:
                    torch.save(attn[:,0,layers,:,:], attn_path)
            
            predicted_token_id = logits.argmax().item()
            predicted_token_str = processor.decode(predicted_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            pred_ans = processor.batch_decode(top_logit)
            
            temp = {
                "id": sample['id'],
                "predicted_ans": pred_ans,
                "real_ans": sample['answer'],
                "attn_path": attn_path
            }
            
            temp['is_correct'] = pred_ans == sample['answer']
                
            out_samples[sample['id']] = temp
            # break
    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None,help='name of saved json')
    parser.add_argument('--noop', type=str, default="text")
    parser.add_argument('--model', type=str, default="llava_one")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--shuffle',action='store_true')  # on/off flag
    parser.add_argument('--return_logits',action='store_true')  # on/off flag
    parser.add_argument('--reprocess',action='store_true')  # on/off flag
    parser.add_argument('--text_input_type', type=str, default="caption", help="caption, statement")
    parser.add_argument('--conflict_type', type=str, default="direct_alternative", help="aligned, direct_alternative, direct_negation, indirect_negation, indirect_alternative")
    parser.add_argument('--debug',action='store_true')  # on/off flag
    parser.add_argument('--subsets', type=int, default=None)
    parser.add_argument('--store_all',action='store_true')  # on/off flag

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join("dataset/shape_dataset/cache", args.model, f"{args.text_input_type}_{args.conflict_type}")
    os.makedirs("results/"+args.output_dir, exist_ok=True)
    return args
    
def process_single_sample(sample):
    if "image_path" in sample and os.path.isfile(sample["image_path"]):
        img = Image.open(sample["image_path"]).convert("RGB")
        sample["image"] = img
    return sample


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

    for idx in range(1,8):
        name = f"image_{idx}"
        images = []
        if name in sample and sample[name] is not None:
            images.append(sample[name])
            
    sample["images"] = []
    conversation = []
    image_placeholder = {"type": "image"}

    pattern = r"<image (?:10|[0-9])>"
    
    question = sample.get("question", "")
    split_question = re.split(pattern, question)
    if sample['question_type'] == "multiple-choice":
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For multiple choice questions, please only reply with a single letter for your answer choice.")
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
    sample["final_input_prompt_base"] = [response]

    
    noop_sample = sample.copy()
    question = noop_sample.get("noop", "")
    split_question = re.split(pattern, question)
    if noop_sample['question_type'] == "multiple-choice":
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For multiple choice questions, please only reply with a single letter for your answer choice.")
        # instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately. "
        #                "please fully explain your reasoning steps behind your answer and end your reply with your final choice in brackets.")
    else:
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For open questions, reply as succinctly as possible with the final answer only.")

    if noop_sample['noop_warning']:
        warn_text = "This might be a trick question designed to confuse LLMs with additional information. Look for irrelevant information or distractors in the question:"
        instruction += "\n" + warn_text

    content = [{"type": "text", "text": instruction}]
    question_dict, question_images = string_to_dict(question)
    content += question_dict
    # sample["images"] += question_images

    if noop_sample['question_type'] == "multiple-choice":
        answers = ast.literal_eval(noop_sample['options'])
        ops = 'ABCDEFGHIJKL'
        content.append({"type": "text", "text": "Your options are: "})
        for i, ans in enumerate(answers):
            content.append({"type": "text", "text": ops[i] + ") "})
            answer_dict, answer_imgs = string_to_dict(ans)
            # sample["images"] += answer_imgs
            content += answer_dict

    response = {
        "role": "user",
        "content": content
    }
    sample["final_input_prompt_noop"] = [response]
    return sample
    
#%%
def main():
    args = get_args()
    device = "cuda:0"
    assert args.conflict_type in ["direct_aligned", "no_conflict", "direct_alternative", "direct_negation", "indirect_negation", "indirect_alternative"]
    assert args.text_input_type in ["caption", "statement"]
    
    print("Loading Dataset")
    print(args.noop)
    dataset = load_dataset(args.noop)
    
    print("Example Datapoint: ")
    print(dataset[0])
         
    print("Loading Model")
    processor, model = load_model(args.model, device)
        
    print("formatting prompts")
    formatted_dataset = []
    for sample in dataset:
        sample['noop_warning'] = False
        new_sample = process_single_sample(sample)
        new_sample = construct_prompt_l1(new_sample)
        formatted_dataset.append(new_sample)
    print("Example formatted sample")
    print(formatted_dataset[0])
    
    print("running model")
    
    
    out_samples = extract_attention(
        formatted_dataset, 
        model, 
        processor, 
        args.output_dir, 
    )
    
    json_path = os.path.join("results", args.output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(out_samples,f)
        
    print("results saved to ", json_path)

    
    
    
if __name__ == "__main__":
    main()
