import re
import ast

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
    question = sample.get("noop", sample.get("question", ""))
    image_names = [f"image_{idx}" for idx in range(1, 8)]
    sample["images"] = []
    if sample.get("question_type") == "multiple-choice":
        instruction = (
            "you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
            "For multiple choice questions, please only reply with a single letter for your answer choice."
        )
    else:
        instruction = (
            "you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
            "For open questions, reply as succinctly as possible with the final answer only."
        )
    if sample.get('noop_warning'):
        warn_text = ("This might be a trick question designed to confuse LLMs with additional information. "
                     "Look for irrelevant information or distractors in the question:")
        instruction += "\n" + warn_text

    content = [{"type": "text", "text": instruction}]
    question_dict, question_images = string_to_dict(question)
    content += question_dict
    sample["images"] += question_images

    if sample.get("question_type") == "multiple-choice":
        answers = ast.literal_eval(sample['options'])
        ops = 'ABCDEFGHIJKL'
        content.append({"type": "text", "text": "Your options are: "})
        for i, ans in enumerate(answers):
            content.append({"type": "text", "text": ops[i] + ") "})
            answer_dict, answer_imgs = string_to_dict(ans)
            sample["images"] += answer_imgs
            content += answer_dict

    response = {"role": "user", "content": content}
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

def replace_image_tokens(text):
    pattern = re.compile(r'<image\s+(\d+)>')
    images_replaced = []
    replacement_count = 0
    def _replacer(match):
        nonlocal replacement_count
        original_number = match.group(1)
        replacement_count += 1
        new_token = f"Image-{replacement_count}<image>"
        images_replaced.append({
            "image_new": new_token,
            "image_old": original_number
        })
        return new_token
    modified_text = pattern.sub(_replacer, text)
    return modified_text, images_replaced

def construct_prompt_intern(sample, load_image_func):
    question = sample.get("noop", sample.get("question", ""))
    if sample.get("question_type") == "multiple-choice":
        instruction = (
            "you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
            "For multiple choice questions, please only reply with a single letter for your answer choice."
        )
    else:
        instruction = (
            "you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
            "For open questions, reply as succinctly as possible with the final answer only."
        )
    op_text = ""
    if sample.get("question_type") == "multiple-choice":
        answers = ast.literal_eval(sample['options'])
        ops = 'ABCDEFGHIJKL'
        op_text += "Your options are:\n"
        for i, ans in enumerate(answers):
            op_text += ops[i] + ") " + ans + "\n"

    response = instruction + "\n" + question + "\n" + op_text
    response, images = replace_image_tokens(response)
    pixel_values = []
    img_sizes = []
    for img in images:
        # load_image_func should load the image and return a tensor (or PIL image)
        img_tensor = load_image_func(sample[f"image_{img['image_old']}"])
        pixel_values.append(img_tensor)
        img_sizes.append(img_tensor.size(0))
    import torch
    sample["final_input_prompt"] = [response]
    sample["pixel_values"] = torch.cat(pixel_values, dim=0)
    sample["image_sizes"] = torch.tensor(img_sizes)
    return sample
