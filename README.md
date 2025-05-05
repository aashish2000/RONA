# RONA: Pragmatically Diverse Image Captioning with Coherence Relations

[Aashish Anantha Ramakrishnan](https://aashish2000.github.io), [Aadarsh Anantha Ramakrishnan](https://www.linkedin.com/in/aadarsh-a/), [Dongwon Lee](https://scholar.google.com/citations?user=MzL-WnEAAAAJ&hl=en)

Accepted in the [4th In2Writing Workshop](https://aclanthology.org/volumes/2025.in2writing-1/), co-located with NAACL 2025.

<p align="left">
  <a href='https://arxiv.org/abs/2503.10997'>
  <img src='https://img.shields.io/badge/Arxiv-2503.10997-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://aclanthology.org/2025.in2writing-1.8.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=yellow'></a> 
</p>

## TL;DR

We propose RONA, a Coherence Relation-based pragmatic prompting strategy for MLLMs. Our approach generates pragmatically diverse captions, improving over existing baselines that only focus on syntax and semantic variations. 

![Arch Diagram](./images/arch_diagram_final.jpg)

## Setup Instructions

### Installing Packages
We recommend using Conda, to create a virtual environment for evaluation. Clone this repository, and use the following commands:
```
conda create -n <your-env> python=3.10 -y
conda activate <your-env>
pip install -r requirements.txt
```

### Setting up MLLMs
We have used the Vertex AI API to access Claude 3.5 Sonnet v2. If you want to setup Google Cloud Access for Claude, please install the gcloud CLI and follow this [guide](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment).

In the case of GPT4o, we used the Azure OpenAI service to create a custom deployment. To create this, please check [this guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource).

### Datasets Used
We used the [Tweet Subtitles](https://connectpolyu-my.sharepoint.com/personal/21038672r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F21038672r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2Faccepted%5Fpaper%5Fdata%2FEMNLP2022%5Fdiscourse%2Fdataset%2Fmultimodal%5Fdiscourse%5Fdataset%2Ezip&parent=%2Fpersonal%2F21038672r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2Faccepted%5Fpaper%5Fdata%2FEMNLP2022%5Fdiscourse%2Fdataset&ga=1) and [ANNA](https://huggingface.co/datasets/aashananth/ANNA) datasets for our experiments. The test set metadata used for our evaluation is present [here](./metadata/).

Please follow the below directory structure for all datasets. Use the same names for the image folder as mentioned below, as the metadata filenames depend on it. 
```
├── tweet_subtitles
|   ├── tweet_subtitles_test_metadata.json
|   └── multimodal_discourse_dataset/
└── anna
    ├── anna_test_metadata.json
    └── test/
```

### Environment Constants
Rename the file `environment.py.example` to `environment.py` and fill all the constants.
- `DATASET_ROOT_DIRS`: This is the location of the test set images and metadata for each dataset.
- `LOCATION`: Location of the Vertex AI API for Claude
- `AZURE_OPENAI_ENDPOINT`: Endpoint URL of the Azure GPT4o Deployment
- `AZURE_OPENAI_API_KEY`: API Key for the Azure GPT4o Deployment
- `AZURE_API_VERSION`: Version used for the Azure OpenAI API

## Generating RONA Captions
```
python eval_mllm.py --model <model> --dataset <dataset> --seed <seed> --with_pair --with_dc
```
This command creates captions using the RONA Prompting Strategy (image-caption pair + coherence relation). If you want to just use images without the captions, omit the `with_pair` flag. If you do not want to use coherence relations, omit the `with_dc` flag. The results of the evaluation will be saved in the `llm_outputs/` folder.

## Calculating Metrics
```
# To calculate metrics for captions generated in all datasets
python results_bench.py --model <model> --all_datasets

# To calculate metrics for captions generated in a particular dataset
python results_bench.py --model <model> --dataset <dataset>

# To calculate metrics for a particular results file
# (add --dc if it was generated using coherence relations)
python results_bench.py --model <model> --dataset <dataset> --results_file <path-to-file>

# Collecting average scores for each dataset
python results_bench.py --model <model> --dataset <dataset> --get_dataset_avg_scores
```
All scores will be saved in the same `llm_outputs/` folder. Collected average scores will be present in the `all_average_scores/` folder.

## Citing
If you find our work useful, please consider citing:
```
@inproceedings{anantha-ramakrishnan-etal-2025-rona,
    title = "{RONA}: Pragmatically Diverse Image Captioning with Coherence Relations",
    author = "Anantha Ramakrishnan, Aashish  and
      Ramakrishnan, Aadarsh Anantha  and
      Lee, Dongwon",
    editor = "Padmakumar, Vishakh  and
      Gero, Katy  and
      Wambsganss, Thiemo  and
      Sterman, Sarah  and
      Huang, Ting-Hao  and
      Zhou, David  and
      Chung, John",
    booktitle = "Proceedings of the Fourth Workshop on Intelligent and Interactive Writing Assistants (In2Writing 2025)",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.in2writing-1.8/",
    pages = "74--86",
    ISBN = "979-8-89176-239-8"
}
```
