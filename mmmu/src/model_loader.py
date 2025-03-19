import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM
from transformers import LlavaNextForConditionalGeneration, LlavaOnevisionForConditionalGeneration

def handle_model_types(model_type, model_id, cache_dir, args, device):
    if model_type == "AutoModelForCausalLM":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        ).to(device)
        processor = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        ).to(device)
        processor = None
    return tokenizer, processor, model

def load_model_and_tokenizer(args, device, MODEL_DICT_LLMs):
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
    elif model_type == "internvl2_5":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
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
        processor = None
    else:
        tokenizer, processor, model = handle_model_types(model_type, model_id, cache_dir, args, device)

    print("Model loaded successfully!")
    return model, tokenizer, processor
