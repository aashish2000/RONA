import evaluate
import os
import argparse
import json
import string
import torch
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from helper import read_json, save_json, write_csv
from environment import DATASET_ROOT_DIRS
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms

# Fix for BLEURT score taking too much VRAM
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# List of flagged images
images_to_exclude = [
    # Safety filter
    'multimodal_discourse_dataset/rDoylPXF7i.jpg', 
    'multimodal_discourse_dataset/rOD2ERXlHS.jpg', 
    'multimodal_discourse_dataset/sSUREsXW4I.jpg', 
    'multimodal_discourse_dataset/teLhbMwEGe.jpg', 
    'multimodal_discourse_dataset/uPq5XEwsx2.jpg', 
    'multimodal_discourse_dataset/ufPXqiZrMa.jpg', 
    'multimodal_discourse_dataset/uw1S4kYFMg.jpg', 
    'multimodal_discourse_dataset/v1YRGuxUHt.jpg', 
    'multimodal_discourse_dataset/vR8xt6KBOt.jpg', 
    'multimodal_discourse_dataset/vfzzINivBI.jpg',
    'test/faf69237c8b33306a2bb9c38c2583abda12f873e1ceb6720e8dd933ca68c459f.jpg', 
    'test/f56b1e326b55cb237a72bd45384649065296003f9710d91c7eb706871d6e033d.jpg', 
    'test/f09899c9e15c262fdcadedbc52c087f6a4bdb98701857a6f94fda1a04cf8ac48.jpg', 
    'test/f64e3d1d20f17e72915e8f84ec5b36497d0b442c69529e4cd3edf1eb475d13c9.jpg', 
    'test/ecd195744a1b0d51ddd2015ec176545e400497dcccd32fbe5416faadc9887db8.jpg', 
    'test/ef5de873c77480c579926e6fed4f1a3261a8a7a62f33a96ac772ccf65e67993f.jpg', 
    'test/f4625abfd61cc1354b324af66bb4e669220c3d544d6a8425d98ad54e8b3e2db8.jpg', 
    'test/f721559673f82bcd1f3ab8e9f045a672aca15405bb83b1a902968c60cb475d63.jpg', 
    'test/f2c98c4afe5db2fbdeafb0a395385b1bd73f16e754255dec61bcb5b7dc2581cd.jpg', 
    'test/fc822361ef3dc901a875ff8ec8d72c7a5c77cd930e92461fd8f2c2935b58acc6.jpg', 
    'test/ebc66e810dd99821415ae89fde302069a31c4b983065da63720ef288abfe6305.jpg', 
    'test/fa41d4ec55b2d63c5f4c910b135c10da2f354de1b3f43d2143f7838572336f2f.jpg', 
    'test/fef7940fe08082054fb9b538917c2e96239ee929b4d2746eb0f9a0fc31b9ca1f.jpg', 
    'test/fa87bb2ab2a8a9e5c40b8476dda1ab06e0ec636bb5c9142248b247191316a39d.jpg', 
    'test/eb1242c38082d00d350b2dff20dc5ac5d366b3d72b4789e396239e174bad73a9.jpg', 
    'test/f14996aeb5797fea43bcf72e8a2c2413bc29c8818d7f33eef0eb4dff659c4079.jpg',
    # Refusal
    'multimodal_discourse_dataset/r1ddIzxxbC.jpg',
    'multimodal_discourse_dataset/tQcg9cfk0X.jpg',
    'test/fc6a200f77ac2f7038c92bdfd345f4369a21f46c50ae93220e7c948ddcdad935.jpg',
    'test/f9516b107e4f91f2115612520fb2ed301043b575540707c2b7c2a1b11012718d.jpg',
    'test/fcf2d386d436e92ded67438cb7ad417d87344e82971ee4d714dedaba389d95a3.jpg',
    'test/f27d37fd5ec30aa2f971ee76f88c2e782d963943463b43c0ebef71f4a4d27444.jpg',
    'test/f3e5139e291d0df6c6da3afd7e3215817ce48050462fb806567e1ef190df9672.jpg',
    'test/ff3c9cce389d377ecfaa2da002f721c331b12c82ccdfb97210ac8a6180c63fce.jpg',
    'test/f476f4589e9c3b6a59d946c819a7ccde52c51869cdfdae18380f911c76fa88b9.jpg',
    'test/ea80e5f1b1e35da3c103815c6588608d1b4b166f3a929b867b1f9a2126aa3a8a.jpg',
    'test/fd1397a25d9c00c3c39a195c202e5672abec7dae1e1f572cfb96d237199c5c31.jpg',
    'test/ff4754837d6014775ab9f809cacc7833b2a7e6fb7675513b5ac3fe69bb214f99.jpg',
    'test/f5f22ae3e5e8c58dee11fc4a2bc28488bd6ebb0faaee39f359717ad3d8795e92.jpg',
    'test/fb49085fcc829f1d909eda46d7b696c9518cd45c88ec3d249eff0f31146772b7.jpg',
    'test/f3a31c67b562a57aaaaef22f1f11f3c3e6ff505170a809c757af2efa0cb22b4a.jpg',
    'test/f706f554d43a0a984a47db6ecd4d91bd62f8dc3523f66a4efe10a4ddb44cda58.jpg',
    'multimodal_discourse_dataset/tc7qiH6eeB.jpg',
    'multimodal_discourse_dataset/u0lCvPhU76.jpg',
    'multimodal_discourse_dataset/uqSG6Y0GYl.jpg',
    'multimodal_discourse_dataset/vLu29g5wUC.jpg',
    'multimodal_discourse_dataset/w9TMjYEiKA.jpg',
    'multimodal_discourse_dataset/xCyTRNITDI.jpg',
    'multimodal_discourse_dataset/y0OST5UKB3.jpg',
    'multimodal_discourse_dataset/yvXfGpjEta.jpg',
    'multimodal_discourse_dataset/qwpGmcibIq.jpg',
    'multimodal_discourse_dataset/xsFdFfmstn.jpg',
    'multimodal_discourse_dataset/yWKsxmbt6O.jpg',
    'multimodal_discourse_dataset/to44zWk6pA.jpg',
]

RESULTS_FOLDER = './llm_outputs/'
AVG_METRICS_FOLDER = './all_average_scores/'

clip_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").cuda()
bleurt = evaluate.load("bleurt", "bleurt-large-512")

def compute_clip_scores(image_path, captions):
    transform_tensor = transforms.ToTensor()
    image = transform_tensor(Image.open(image_path))
    image = image.cuda()

    with torch.no_grad():
        scores = clip_model(image.repeat(len(captions), 1, 1, 1), captions).item()
    
    return scores

def calculate_mbleurtscore(generated_captions):
    bleurt_scores_all = []
    for i in range(len(generated_captions)):
        candidate = [generated_captions[i] for i in range(len(generated_captions) - 1)]
        references = generated_captions[:i] + generated_captions[i+1:]
        
        result = bleurt.compute(predictions=candidate, references=references)
        bleurt_scores_all.append((float(sum(result['scores'])) / 4) * 100) 

    average_bleurt = np.mean(bleurt_scores_all) / 100  # Normalize to [0,1]

    return average_bleurt

def calculate_div2(captions):
    all_words = []
    all_bigrams = set()
    
    for caption in captions:
        words = nltk.word_tokenize(caption.lower())  # Tokenize words
        all_words.extend(words)
        bigrams = list(nltk.bigrams(words))  # Generate bigrams
        all_bigrams.update(bigrams)  # Store unique bigrams

    num_unique_bigrams = len(all_bigrams)
    num_total_words = len(all_words)

    div2_score = num_unique_bigrams / num_total_words if num_total_words > 0 else 0
    return div2_score

def compute_scores(data):
    all_results = [["filename", "bleurtscore", "mbleurtscore", "clip", "div2"]]
    average_scores = {"bleurtscore": 0, "mbleurtscore": 0, "clip": 0, "div2": 0}
    
    for item in tqdm(data):
        if (item["filename"] in images_to_exclude):
            continue
        reference = [item["caption"]]
        generated_captions = item["generated_captions"]
        
        # Handle dictionary or list format
        if isinstance(generated_captions, dict):
            generated_captions = list(generated_captions.values())
        
        # Clean all generated captions
        printable = set(string.printable)
        generated_captions = [''.join(filter(lambda x: x in printable, s)) for s in generated_captions]

        # Compute BLEURT score
        bleurt_score = bleurt.compute(predictions=generated_captions, references=[reference[0] for i in range(len(generated_captions))])

        # Compute MBLEURT score
        mbleurt_score = calculate_mbleurtscore(generated_captions)

        # Compute CLIP score
        if(item["filename"].startswith("test")):
            image_path = DATASET_ROOT_DIRS['anna'] + item["filename"]
        else:
            image_path = DATASET_ROOT_DIRS['tweets_sl'] + item["filename"]

        clip_score = compute_clip_scores(image_path, generated_captions)

        # Compute diversity score
        div2_score = calculate_div2(generated_captions)
            
        # Append results
        all_results.append([
            item["filename"],
            float(sum(bleurt_score['scores'])) / 4,
            mbleurt_score,
            clip_score,
            div2_score,
        ])

        # Update average scores
        average_scores["bleurtscore"] += float(sum(bleurt_score['scores'])) / 4
        average_scores["mbleurtscore"] += mbleurt_score
        average_scores["clip"] += clip_score
        average_scores["div2"] += div2_score

    # Compute average scores
    average_scores["bleurtscore"] /= len(data)
    average_scores["mbleurtscore"] /= len(data)
    average_scores["clip"] /= len(data)
    average_scores["div2"] /= len(data)

    return all_results, average_scores

def eval_and_save(data_file, per_example_file, average_file):
    data = read_json(data_file)
    all_results, average_scores = compute_scores(data)

    write_csv(per_example_file, all_results)
    save_json(average_scores, average_file)
    print(f"Results for {data_file}:")
    print(json.dumps(average_scores, indent=4))
    print()

def combine_all_average_scores(folder, model, dataset):
    df = []
    for dc in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, dc)):
            if file.startswith("average_"):
                average_file = os.path.join(folder, dc, file)
                file_data = {}
                file_data["dc"] = True if "no_dc" not in dc else False
                file_data["pair"] = True if "no_pair" not in file else False

                data = read_json(average_file)
                data = {**file_data, **data}
                df.append(data)

    df = pd.DataFrame(df)
    df.to_csv(AVG_METRICS_FOLDER + f"{model}_{dataset}_all_avg_scores.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics for VLM models.')
    parser.add_argument('--model', type=str, help='Model to evaluate', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate')
    parser.add_argument('--all_datasets', action='store_true', help='Evaluate all datasets')
    parser.add_argument('--dc', action='store_true', help='Were results generated with Coherence Relations?')
    parser.add_argument('--results_file', metavar='text', nargs='?', help='Results file')
    parser.add_argument('--get_dataset_avg_scores', action='store_true', help='Get all average scores for a dataset')

    args = parser.parse_args()
    if (args.get_dataset_avg_scores):
        target_folder = os.path.join(RESULTS_FOLDER, args.model, args.dataset)
        combine_all_average_scores(target_folder, args.model, args.dataset)
        exit(0)

    if (args.all_datasets):
        target_folder = os.path.join(RESULTS_FOLDER, args.model)
        for dataset in os.listdir(target_folder):
            for folder in os.listdir(os.path.join(target_folder, dataset)):
                for file in os.listdir(os.path.join(target_folder, dataset, folder)):
                    if file.endswith(".json") and file.startswith("results"):
                        print(f"Processing {file} in {dataset}/{folder}")
                        eval_and_save(
                            os.path.join(target_folder, dataset, folder, file), 
                            os.path.join(target_folder, dataset, folder, "per_example_"+file.replace(".json", ".csv")), 
                            os.path.join(target_folder, dataset, folder, "average_"+file)
                        )
    else:
        target_folder = os.path.join(RESULTS_FOLDER, args.model, args.dataset)
        if (not args.results_file): 
            for folder in os.listdir(target_folder):
                for file in os.listdir(os.path.join(target_folder, folder)):
                    if file.endswith(".json") and file.startswith("results"):
                        print(f"Processing {file} in {folder}")
                        eval_and_save(
                            os.path.join(target_folder, folder, file), 
                            os.path.join(target_folder, folder, "per_example_"+file.replace(".json", ".csv")), 
                            os.path.join(target_folder, folder, "average_"+file)
                        )
        else:
            if (args.dc):
                target_folder = os.path.join(target_folder, 'dc')
            else:
                target_folder = os.path.join(target_folder, 'no_dc')
            
            print(f"Processing {args.results_file}")
            eval_and_save(
                os.path.join(target_folder, args.results_file), 
                os.path.join(target_folder, "per_example_"+args.results_file.replace(".json", ".csv")), 
                os.path.join(target_folder, "average_"+args.results_file)
            )