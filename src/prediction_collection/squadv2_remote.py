from typing import List

import numpy as np
import pandas as pd

from src.prediction_collection._gpt3_access import get_gpt3_answers, GPT3Answer
from src.prediction_collection.squadv2_utils import get_squad_dataset

request_template = """
Title: {title}
Background: {background}
Question: {practice_question}
Answer (extraction from background, as short as possible): {practice_answer}
Question: {question}
Answer (extraction from background, as short as possible):

"""

NUM_SAMPLES = 10866  # all of them


def collect_remote_predictions():
    test_set = get_squad_dataset()

    test_set = test_set[:NUM_SAMPLES]

    prompts = [request_template.format(title=sample.title,
                                       background=sample.background,
                                       practice_question=sample.practice_question,
                                       practice_answer=sample.practice_answer,
                                       question=sample.question)
               for sample in test_set]

    answers: List[GPT3Answer] = get_gpt3_answers(prompts=prompts,
                                                 model="text-davinci-003",
                                                 max_tokens=20)

    res_df = pd.DataFrame(columns=['id', 'exact_match', 'is_possible', 'gpt3_confidence', 'pred_time'])
    for sample, answer in zip(test_set, answers):
        answer_text = answer.answer['choices'][0]['text'].strip()
        em = sample.is_exact_match(answer_text)
        print(f"EM: {em}, Impossible: {sample.is_impossible}  Answer: {answer_text}, Correct Answers: {sample.answers}")

        token_scores = np.exp(answer.answer['choices'][0]['logprobs']['token_logprobs'])

        supervisor_score = float(np.min(token_scores))
        res_df = res_df.append({'id': sample.id,
                                'is_possible': not sample.is_impossible,
                                'exact_match': em,
                                'gpt3_confidence': supervisor_score,
                                'pred_time': answer.time}, ignore_index=True)

    res_df.to_csv(f"/generated/predictions_and_uncertainties/squadv2_remote.csv", index=False)


if __name__ == '__main__':
    collect_remote_predictions()
