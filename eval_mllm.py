import configs
import os
import json
import argparse
from prompt_builder import build_prompt_for_mllm
from environment import DATASET_ROOT_DIRS
from helper import read_json, save_json, clean_json_string

images_with_errors = []

def call_model(model):
    if (model == 'gpt4o'):
        from load_models import load_gpt4o, call_gpt4o
        client = load_gpt4o()
        return lambda configs: call_gpt4o(
            client,
            **configs
        )
    elif (model == 'claude'):
        from load_models import load_claude, call_claude
        client, creds = load_claude()
        return lambda configs: call_claude(
            client,
            creds,
            **configs
        )
    else:
        raise NotImplementedError(f"Unknown model: {model}")

def eval_mllm(model, dataset, results_file, with_pair, with_dc, seed = 42):
    data = read_json(configs.test_metadata_paths[dataset])
    if(os.path.exists(results_file)):
        results = read_json(results_file)
    else:
        results = []
    
    total = len(data)
    for i, item in enumerate(data):
        print()
        print(f"{i+1}/{total}")

        caption = item["caption"]
        filename = item["filename"]
        image_path = os.path.join(DATASET_ROOT_DIRS[dataset], filename)

        if (i < len(results)):
            continue

        if (with_pair):
            messages = build_prompt_for_mllm(
                configs.DC_SYSTEM_MESSAGE if with_dc else configs.WITHOUT_DC_SYSTEM_MESSAGE,
                configs.PROMPT_DC_WITH_PAIR if with_dc else configs.PROMPT_WITHOUT_DC_WITH_PAIR,
                image=image_path,
                caption=caption,
                image_mode = "path",
                image_input_detail = "high",
                claude_inference=(model == 'claude')
            )
        else:
            messages = build_prompt_for_mllm(
                configs.DC_SYSTEM_MESSAGE if with_dc else configs.WITHOUT_DC_SYSTEM_MESSAGE,
                configs.PROMPT_DC_WITH_IMAGE if with_dc else configs.PROMPT_WITHOUT_DC_WITH_IMAGE,
                image=image_path,
                caption=None,
                image_mode = "path",
                image_input_detail = "high",
                claude_inference=(model == 'claude')
            )
        
        call_mllm = call_model(model)

        system_msg = None
        if (model == 'claude'):
            # No system message for Claude
            system_msg = messages[0]['content'][0]['text']
            messages = messages[1:]

        params = {"prompt": messages}
        if (system_msg):
            params['system_msg'] = system_msg
        else:
            params['seed'] = seed
        
        try:
            response = call_mllm(params)
        except Exception as e:
            images_with_errors.append((filename, e))
            continue

        print (response)
        if (with_dc):
            response_cleaned = clean_json_string(response, type_of_json="object")
        else:
            response_cleaned = clean_json_string(response, type_of_json="array")

        generated_captions = json.loads(response_cleaned)
        if (not generated_captions):
            images_with_errors.append((filename, "Refusal/Wrong format"))
        
        result = {
            "filename": filename,
            "caption": caption,
            "generated_captions": generated_captions
        }
        results.append(result)
        save_json(results, results_file)

################################################## Invocation #####################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MLLMs for Image Captioning with Coherence Relations')
    parser.add_argument('--model', type=str, help='Model to evaluate', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', required=True, choices=['tweets_sl', 'anna'])
    parser.add_argument('--seed', type=int, help='Random seed', required=True)
    parser.add_argument('--results_file', metavar='text', nargs='?', help='Results file')
    parser.add_argument('--with_pair', action='store_true', help='Evaluate with image-caption pair')
    parser.add_argument('--with_dc', action='store_true', help='Evaluate with Discourse Coherence Relations')
    args = parser.parse_args()

    results_path = f'./llm_outputs/{args.model}/{args.dataset}/'
    if (args.with_dc):
        results_path = os.path.join(results_path, 'dc/')
    else:
        results_path = os.path.join(results_path, 'no_dc/')

    os.makedirs(results_path, exist_ok=True)
    
    with_pair_str = 'with_pair' if args.with_pair else 'no_pair'

    if (args.results_file): 
        results_path = os.path.join(results_path, args.results_file)
    else:
        results_path = os.path.join(results_path, f'results_{with_pair_str}.json')
    
    eval_mllm(
        model = args.model,
        dataset = args.dataset,
        results_file = results_path,
        with_pair = args.with_pair,
        with_dc = args.with_dc,
        seed = args.seed
    )
    print (images_with_errors)