import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_openai(image_input, image_mode, image_input_detail):
    if image_mode == 'url':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'path':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_input)}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {image_mode}.")
    
    return image_content

def process_image_anthropic(image_input, image_mode, image_input_detail):
    # image_mode and image_input_detail are not used in this function
    image_content = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": encode_image(image_input),
        },
    }
    
    return image_content

def process_text(text_input):
    text_content = {
        "type": "text",
        "text": text_input,
    }
    return text_content

def build_prompt_for_mllm(system_msg, instruction, image, caption, image_mode, image_input_detail, claude_inference=False):
    if claude_inference:
        process_image = process_image_anthropic
    else:
        process_image = process_image_openai

    messages = []
    # System message
    messages.append({
        "role": "system",
        "content": [process_text(system_msg)],
    })
    # Task prompt
    task_prompt = {
        "role": "user",
        "content": [process_text(instruction[0])],
    }
    
    task_prompt["content"][0]['text'] += instruction[1] + instruction[2]

    messages.append(task_prompt)

    # Add image or image-caption pair
    content = [process_image(image, image_mode, image_input_detail)]
    if caption is not None:
        content.append(process_text(caption))

    messages.append({
        "role": "user",
        "content": content,
    })
    return messages