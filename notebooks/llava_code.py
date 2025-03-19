from transformers import AutoProcessor
import torch

# TODO: Implement the following functions
def get_token_position(conversation):
    """
    Return token position of instruction, image, question, and predicted answer.
    Make it easier for exploratory analysis

    image_token = 151646 for llava-onevision-qwen-0.5b
    model.config.image_token_index = 151646
    """
    pass


def map_token_to_pixel():
    """
    Return the mapping of each visual token to the pixel location in the image.
    """
    pass

def load_model_llava(model_path, device, dtype=torch.float16):
    if "onevision" in model_path:
        from transformers import LlavaNextForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoProcessor
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        
    elif "llava-v1.6-mistral" in model_path:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            ).to(device)
        
    elif "llava-1.5" in model_path:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            ).to(device)
    return model, processor

def format_prompt_llava(sample, text_input_type="caption", conflict_type="direct_alternative"):
    #TODO: need to refactor to be dataset-agnostic
    instruction = f'''
    You are an advanced visual question answering vision language model.
    You have the task of answering my question about an image {text_input_type} pair.
    First I will provide the image, then the {text_input_type}, and finally my question.
    Answer the question with a single word if possible, more than a single word will be scored as incorrect.
    '''
    question = f"what color is the {sample['params']['shape']}"
    caption = sample['caption']
    content = []
    content.append({"type": "text", "text": instruction})
    if text_input_type == "statement":
        if conflict_type == "direct_alternative":
            statement = f"The {sample['params']['shape']} is {sample['params']['text_color']}."
        elif conflict_type == "direct_negation":
            statement = f"The {sample['params']['shape']} is not {sample['params']['image_color']}."
        elif conflict_type == "indirect_negation":
            statement = f"All {sample['params']['shape']} are not {sample['params']['image_color']}."
        elif conflict_type == "indirect_alternative":
            statement = f"All {sample['params']['shape']} are {sample['params']['text_color']}."
        else:
            raise ValueError(f"Conflict type {conflict_type} not recognized")
        content.append({"type": "text", "text": statement})
    else:
        if conflict_type == "direct_aligned":
            caption = f"an image of a {sample['params']['image_color']} {sample['params']['shape']}"
            sample['params']['text_color'] = sample['params']['image_color']
        elif conflict_type == "no_conflict":
            caption = f"an image of a {sample['params']['shape']}"
            sample['params']['text_color'] = "none"
        else:
            caption = sample['caption']
        content.append({"type": "text", "text": caption})
    content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    
    
    sample['text_input_type'] = text_input_type
    sample['conflict_type'] = conflict_type
    sample['text_input'] = content
    sample['image_input'] = [sample['image']]
    
    return sample

def prepare_inputs_llava(sample, processor, device):
    final_input_prompt = [{
        "role": "user",
        "content": sample['text_input']
    }]
    final_input_prompt = processor.apply_chat_template(final_input_prompt, add_generation_prompt=True)
    inputs = processor(
        text=final_input_prompt,
        images=sample['image_input'],
        return_tensors="pt",
        padding=True
    )
    return inputs.to(device)

def call_engine_llava(args, sample, model, processor, return_logits=False):
    device = next(model.parameters()).device

    inputs = prepare_inputs_llava(sample, processor, device)

    # TODO: refactor to use https://huggingface.co/docs/transformers/v4.48.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput
    # This allows printing both logits and normalized logit.
    if not return_logits:    
        generation_config = {
            "max_new_tokens": 256,            # Limit the number of tokens generated
            "temperature": 1.0,               # Neutral temperature; does not affect greedy decoding
            "do_sample": False,               # Disable sampling
            "num_beams": 1,                   # Greedy decoding (no beam search)
            "repetition_penalty": 1.0,        # No penalty for repeated tokens
            "length_penalty": 1.0,            # No length bias
            "early_stopping": True,           # Stop as soon as EOS is reached
            "pad_token_id":processor.tokenizer.eos_token_id
        }

        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, **generation_config, 
            )
            predicted_token_id = generate_ids[0, inputs['input_ids'].shape[1]:]
            predicted_token_str = processor.batch_decode(
                predicted_token_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return {
                "predicted_token_id": predicted_token_id.tolist(),
                "predicted_token_str": predicted_token_str
            }
    
    else:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1] # get the logit of the last token
            # get the predicted token using argmax
            predicted_token_id = logits.argmax().item()
            predicted_token_str = processor.decode(predicted_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {
            "predicted_token_id": [predicted_token_id],
            "predicted_token_str": predicted_token_str,
            "logits": logits,
            "probits": torch.nn.functional.softmax(logits)
        }

