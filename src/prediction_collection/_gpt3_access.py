import json
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import List, Dict

import openai
from filelock import FileLock

try:
    with open("/generated/gpt3/access_token.txt", "r") as f:
        openai.api_key = f.read().strip()
except FileNotFoundError:
    pass


@dataclass
class GPT3Answer:
    model: str
    prompt: str
    answer: str
    time: float
    max_tokens: int


def get_gpt3_answers(prompts: List[str], model: str, max_tokens: int = 1, hash_prompt: bool = False) -> List[
    GPT3Answer]:
    """Get a prediction from the GPT-3 API."""
    answers = []

    some_requests_failed = False

    # Open cache pickle
    cache_file = f"/generated/gpt3/{model}.json"
    lock = FileLock(f"{cache_file}.lock")
    with lock:
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except FileNotFoundError:
            cache: Dict[str, str] = dict()

        for i, prompt in enumerate(prompts):

            if hash_prompt:
                # Due to copyright, for some datasets we cannot store the prompt in the cache.
                # Instead, we hash it and use the hash as the key and prompt value.
                #   This works as we never need to read the prompt from the cache.
                cache_key = sha256(prompt.encode('utf-8')).hexdigest()
            else:
                cache_key = prompt

            try:
                answer = cache[cache_key]
                answer = json.loads(answer)
                assert answer["max_tokens"] == max_tokens, "Cache entry has wrong max_tokens"
                answers.append(GPT3Answer(
                    model=answer["model"],
                    prompt=prompt,
                    answer=answer["answer"],
                    time=float(answer["time"]),
                    max_tokens=int(answer["max_tokens"]))
                )
                print(f"Cache hit for prompt {i}...")
            except KeyError:
                # Make API call
                start = time.time()

                try:
                    res = openai.Completion.create(
                        engine=model,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=max_tokens,
                        top_p=1,
                        logprobs=5,
                    )
                except openai.error.AuthenticationError as e:
                    print("No valid API key found. Please set the OPENAI_API_KEY environment variable"
                          "or add a file `generated/gpt3/access_token.txt` containing just the key.")
                except openai.error.ServiceUnavailableError as e:
                    print("GPT-3 API is currently unavailable. "
                          "Will pause for a bit and then continue."
                          "Attention: The script will have to be restarted manually, "
                          "to attempt again the failed approaches")
                    some_requests_failed = True
                    time.sleep(20)
                    print("Continuing...")
                    continue
                except openai.error.InvalidRequestError as e:
                    print(f"Invalid request. Skipping prompt {i}...")
                    some_requests_failed = True
                    continue

                end = time.time()

                gpt_answer = GPT3Answer(model=model,
                                        prompt=cache_key,
                                        answer=res,
                                        time=end - start,
                                        max_tokens=max_tokens)
                answers.append(gpt_answer)
                print(f"Got GPT-3 answer for prompt {i}...")

                # Update cache
                cache[cache_key] = json.dumps(gpt_answer.__dict__)

                with open(cache_file, "w") as f:
                    json.dump(cache, f)

    if some_requests_failed:
        raise RuntimeError("Some requests failed due to overloaded openAI servers."
                           "The script continued after some waiting, but the failed requests"
                           "need to be attempted again: Please restart the script."
                           ""
                           "Note: Thanks to caching, successful requests will not be repeated.")

    return answers
