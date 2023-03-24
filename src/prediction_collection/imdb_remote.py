import time
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from src.prediction_collection._gpt3_access import get_gpt3_answers, GPT3Answer

request_template = """Please classify the following movie reviews into the following sentiments: Negative, Positive.
---
Text: I hate this movie
Label: Negative
---
Text: I like this movie
Label: Positive
---
Text: {}
Label:"""

NUM_SAMPLES = 25000


def sanitize_text(text):
    return text.replace('\n', ' ').replace('"', '\\"')


def _is_negative_token(string: str) -> bool:
    """Check if a token is a negative token."""
    string = string.strip().lower()
    return string in {
        "negative",
        "no",
        "Neg",
        "Hate",
        "Dislike"
    }


tokenizer = None


def _shorten_prompt(prompt: str, max_length: int) -> str:
    global tokenizer
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    original_length = len(tokenizer(prompt)['input_ids'])

    while True:
        number_of_tokens = len(tokenizer(prompt)['input_ids'])
        if number_of_tokens <= max_length:
            if number_of_tokens < original_length:
                print(f"Shortened prompt from {original_length} to {number_of_tokens} tokens.")
            return prompt
        else:
            # Remove last 10 digits
            prompt = prompt[:-10]


def _is_positive_token(string: str) -> bool:
    """Check if a token is a positive token."""
    string = string.strip().lower()
    return string in {
        "positive",
        "yes",
        "Pos",
        "Like",
        "Love"
    }


def _logprobs_to_softmax(logprob: float) -> float:
    """Convert logprob to softmax."""
    return float(np.exp(logprob))


def collect_remote_predictions():
    test_set = load_dataset('imdb', split='test')

    test_set = test_set[:NUM_SAMPLES]

    prompts = [request_template.format(sanitize_text(x)) for x in test_set['text']]

    prompts = [_shorten_prompt(x, max_length=1995) for x in tqdm(prompts, desc="Shortening prompts")]

    answers: List[GPT3Answer] = get_gpt3_answers(prompts=prompts,
                                                 model="text-curie-001",
                                                 max_tokens=20,
                                                 hash_prompt=True)

    predictions = []
    confidences = []
    times = []
    superv_times = []

    answer: GPT3Answer
    for answer in answers:
        pred_token_value = answer.answer['choices'][0]['text'].strip()
        pred, equivalence_checker = pred_label_from_token(pred_token_value)

        start_time = time.time()
        confidence = 0
        for token, logprob in answer.answer['choices'][0]['logprobs']['top_logprobs'][0].items():
            if equivalence_checker(token):
                confidence += _logprobs_to_softmax(logprob)
        superv_times.append(time.time() - start_time)

        predictions.append(pred)
        confidences.append(confidence)
        times.append(answer.time)

    res_frame = pd.DataFrame({'id': list(range(len(predictions))),
                              'prediction': predictions,
                              'ground_truth': test_set['label'],
                              'sm_confidence': confidences,
                              'pred_time': times,
                              'supervisor_time': superv_times})

    res_frame.to_csv(f"/generated/predictions_and_uncertainties/imdb_remote.csv", index=False)


def pred_label_from_token(pred_token_value):
    if _is_negative_token(pred_token_value):
        return 0, _is_negative_token
    elif _is_positive_token(pred_token_value):
        return 1, _is_positive_token
    else:
        return -1, lambda x: False


if __name__ == '__main__':
    collect_remote_predictions()
