import torch
from tqdm import tqdm
from config import MODEL_DICT_LLMs
from prompt import parse_llava_1v, construct_prompt_intern
from utils import save_json

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
    img_mask = inputs.input_ids.cpu() == IMAGE_TOKEN
    img_mask = img_mask[0]
    txt_mask = torch.tensor([not el for el in img_mask])
    # txt_mask = img_mask

    if args.store_attention:
        with torch.no_grad():
            response = model(
                **inputs, output_attentions=True
            )
            # print(response.keys())
            logits = response.logits.detach().cpu()
            attn = torch.stack([el.detach().cpu() for el in response.attentions])
            attn = attn.detach().cpu()
            img_attn = attn[:,:,:,img_mask,:][:,:,:,:,img_mask]
            txt_attn = attn[:,:,:,txt_mask,:][:,:,:,:,txt_mask]
            attn = [img_attn.mean(dim=[1,3,4]), txt_attn.mean(dim=[1,3,4])]
            predicted_token_id = logits[0, -1].argmax().item()
            predicted_token_str = processor.decode(predicted_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return predicted_token_str, attn

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

def call_internvl2_5_engine(args, sample, model, tokenizer):
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        response = model.chat(
            tokenizer, 
            sample['pixel_values'].to(device=device, dtype=dtype), 
            sample.get("question", ""),
            generation_config,
            num_patches_list=sample["image_sizes"].to(device=device, dtype=torch.int),
            return_history=False
        )
    return [response]

def run_inference(dataset, model, processor, args, tokenizer=None):
    model_type = MODEL_DICT_LLMs[args.model_name].get("model_type", "llava_one")
    samples = []
    print("Processing Dataset")
    for sample in tqdm(dataset):
        sample['noop_warning'] = args.warn_noop
        if model_type == "llava_one":
            from utils import process_single_sample, construct_prompt
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)
            samples.append(sample)
        elif model_type == "internvl2_5":
            from prompt import construct_prompt_intern
            # Pass your image-loading function here (replace lambda as needed)
            sample = construct_prompt_intern(sample, load_image_func=lambda x: x)
            samples.append(sample)
    out_samples_eval_only = {}
    out_samples_parse_and_eval = {}
    model.eval()

    print("Running inference on MMMU dataset...")
    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
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

            sample_id = sample.get("id", str(hash(sample.get("question", ""))))
            subject = "_".join(sample.get("id").split("_")[1:-1]) if "id" in sample else "unknown"
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
    import os
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
