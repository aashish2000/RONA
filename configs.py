from environment import DATASET_ROOT_DIRS

test_metadata_paths = {
    'tweets_sl' : DATASET_ROOT_DIRS['tweets_sl']+'tweet_subtitles_test_metadata.json',
    'anna'      : DATASET_ROOT_DIRS['anna']+'anna_test_metadata.json',
}

DC_SYSTEM_MESSAGE = (
    "You are an expert linguist, and your task is to write image captions with the help of Coherence Relations.\n"
    "A coherence relation describes the structural, logical, and purposeful relationships between an image and its caption, capturing the author’s intent.\n"
    "These are the possible coherence relations that can assigned to an image-caption pair:\n"
    "Insertion: The salient object described in the image is not explicitly mentioned in the caption.\n"
    "Concretization: Both the caption and image contain a mention of the main visual entity.\n"
    "Projection: The main entity mentioned in the caption is implicitly related to the visual objects present in the image.\n"
    "Restatement: The caption directly describes the image contents.\n"
    "Extension: The image expands upon the story or idea in the caption, presenting new elements or elaborations, effectively filling in narrative gaps left by the caption.\n"
)

WITHOUT_DC_SYSTEM_MESSAGE = "You are an expert linguist, and your task is to write image captions."

PROMPT_DC_WITH_IMAGE = [
    "You will be given an image ",
    "as input. Write 5 image captions, one for each coherence relation as your output.\n", 
    """Return the captions as a JSON object with the following format:
    {
        "Insertion": "<insert-caption-text-1>",
        "Concretization": "<insert-caption-text-2>",
        "Projection": "<insert-caption-text-3>",
        "Restatement": "<insert-caption-text-4>",
        "Extension": "<insert-caption-text-5>"
    }
    """
]

PROMPT_DC_WITH_PAIR = [
    "You will be given an image-caption pair ",
    (
        "as input. Write 5 image captions, one for each coherence relation as your output. "
        "Create diverse captions while retaining the same overall meaning of the original image-caption pair.\n"
    ),
    """Return the captions as a JSON object with the following format:
    {
        "Insertion": "<insert-caption-text-1>",
        "Concretization": "<insert-caption-text-2>",
        "Projection": "<insert-caption-text-3>",
        "Restatement": "<insert-caption-text-4>",
        "Extension": "<insert-caption-text-5>"
    }
    """
]

PROMPT_WITHOUT_DC_WITH_IMAGE = [
    "You will be given an image ",
    "as input. Analyze the image and write 5 suitable captions that are diverse, but relevant.\n",
    """Return the captions as a JSON Array with the following format: ["<insert-caption-text-1>", "<insert-caption-text-2>", "<insert-caption-text-3>", "<insert-caption-text-4>", "<insert-caption-text-5>"]"""
]

PROMPT_WITHOUT_DC_WITH_PAIR = [
    "You will be given an image-caption pair ",
    (
        "as input. Analyze the image and write 5 suitable captions that are diverse, but relevant. "
        "Create diverse captions while retaining the same overall meaning of the original image-caption pair.\n"
    ),
    """Return the captions as a JSON Array with the following format: ["<insert-caption-text-1>", "<insert-caption-text-2>", "<insert-caption-text-3>", "<insert-caption-text-4>", "<insert-caption-text-5>"]"""
]