import time

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from src.prediction_collection.squadv2_utils import get_squad_dataset, SquadV2TestSample


def collect_local_predictions():
    test_set = get_squad_dataset()

    qa_pipeline = pipeline(
        "question-answering",
        model="mrm8488/bert-tiny-5-finetuned-squadv2",
        tokenizer="mrm8488/bert-tiny-5-finetuned-squadv2"
    )

    exact_matches = []
    sm_confidence = []
    pred_times = []
    possible = []

    sample: SquadV2TestSample
    for sample in tqdm(test_set, desc="Collecting local squadv2 predictions"):
        start_time = time.time()
        answer = qa_pipeline({
            'context': sample.background,
            'question': sample.question
        })
        exact_matches.append(sample.is_exact_match(answer['answer']))
        sm_confidence.append(answer['score'])
        pred_times.append(time.time() - start_time)
        possible.append(not sample.is_impossible)

    res_frame = pd.DataFrame({'index': list(range(len(exact_matches))),
                              'is_possible': possible,
                              'exact_match': exact_matches,
                              'sm_confidence': sm_confidence,
                              'pred_time': pred_times})

    res_frame.to_csv(f"/generated/predictions_and_uncertainties/squadv2_local.csv", index=False)


if __name__ == '__main__':
    collect_local_predictions()
