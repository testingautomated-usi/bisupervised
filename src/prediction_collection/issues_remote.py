# Code largely copy-pasted from https://github.com/MalihehIzadi/catiss/ (MIT licensed)
import os
import os.path
import re
import time
import unicodedata as ud
import warnings
from functools import partial

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from tqdm import tqdm

from src.prediction_collection.issues_utils import prepare_datasets

prepare_datasets()


os.environ["CUDA_VISIBLE_DEVICES"] = ""


NUM_SAMPLES = 10000

def preprocess_inputs():
    test = pd.read_csv("/generated/datasets/issues-test/github-labels-top3-803k-test.csv")

    label = 'issue_label'
    time = 'issue_created_at'
    repo = 'repository_url'
    title = 'issue_title'
    body = 'issue_body'
    author = 'issue_author_association'
    label_col = 'labels'
    text_col = 'text'
    max_title = 30
    max_body = 170
    punctuations = '!"$%&\()*,/:;<=>[\\]^`{|}~+#@-`'
    issue_regex = re.compile(r'#[0-9]+')
    function_regex = re.compile(r'[a-zA-Z][a-zA-Z0-9_.]*\([a-zA-Z0-9_, ]*\)')
    ascii_regex = re.compile(r'[^\x00-\x7f]')

    test[title] = test[title].astype(str)
    test[body] = test[body].astype(str)
    test[author] = test[author].astype(str)
    test[time] = test[time].astype(str)
    test[repo] = test[repo].astype(str)

    # Normalize text
    test[body] = test[body].apply(lambda x: function_regex.sub(" function ", x))
    test[title] = test[title].apply(lambda x: issue_regex.sub(" issue ", x))
    test[body] = test[body].apply(lambda x: issue_regex.sub(" issue ", x))
    test[title] = test[title].str.lower()
    test[body] = test[body].str.lower()

    # Remove extra information
    replace_string = ' ' * len(punctuations)
    test[title] = test[title].str.translate(str.maketrans(punctuations, replace_string))
    test[body] = test[body].str.translate(str.maketrans(punctuations, replace_string))
    test[title] = test[title].apply(lambda x: re.sub(ascii_regex, '', x))
    test[title] = test[title].apply(lambda x: ud.normalize('NFD', x))
    test[body] = test[body].apply(lambda x: re.sub(ascii_regex, '', x))
    test[body] = test[body].apply(lambda x: ud.normalize('NFD', x))
    test[repo] = test[repo].apply(lambda x: x.replace('https://api.github.com/repos/', ''))
    test[title] = test[title].apply(lambda x: " ".join(x.split()))
    test[body] = test[body].apply(lambda x: " ".join(x.split()))

    # truncate columns
    test[title] = test[title].apply(lambda x: ' '.join(x.split(maxsplit=max_title)[:max_title]))
    test[body] = test[body].apply(lambda x: ' '.join(x.split(maxsplit=max_body)[:max_body]))

    # prepare label column for the model
    test[label] = pd.Categorical(test[label])
    test[label_col] = test[label].cat.codes

    # concat issue columns in one "text" column to feed the model
    test[text_col] = 'time ' + test[time] + ' author ' + test[author] + ' repo ' + test[repo] + ' title ' + test[
        title] + ' body ' + test[body]

    return test


def _load_model():
    if not os.path.exists("/generated/trained_models/catiss.bin"):
        warnings.warn("CatISS model not found. "
                      "Please download it from `https://drive.google.com/drive/folders/1jgV4U41-2acctpc6jH5DWL3fF5V6bKF8`"
                      "and place it in `generated/trained_models/catiss.bin`")
        exit(1)

    weights = torch.load("/generated/trained_models/catiss.bin",
                         map_location=torch.device('cpu')
                         )

    from simpletransformers.classification import ClassificationModel, ClassificationArgs

    model_args = ClassificationArgs()
    model_name = 'roberta'
    model_version = 'roberta-base'
    model_args.eval_batch_size = 5
    model_args.max_seq_length = 200
    model_args.n_gpu = 2
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.preprocess_inputs = True

    model = ClassificationModel(model_name,
                                model_version,
                                args=model_args,
                                num_labels=3,
                                use_cuda=False)

    model.model.load_state_dict(weights)

    return model


def _reproduce_catiss_results(model, test):
    def calc(p1, p2, func, **kwargs):
        return func(p1, p2, **kwargs)

    metrics_recom = {
        "accuracy": partial(calc, func=sklearn.metrics.accuracy_score),
        "p_micro": partial(calc, func=sklearn.metrics.precision_score, average='micro'),
        "p_macro": partial(calc, func=sklearn.metrics.precision_score, average='macro'),
        "p_w": partial(calc, func=sklearn.metrics.precision_score, average='weighted'),
        "r_micro": partial(calc, func=sklearn.metrics.recall_score, average='micro'),
        "r_macro": partial(calc, func=sklearn.metrics.recall_score, average='macro'),
        "r_w": partial(calc, func=sklearn.metrics.recall_score, average='weighted'),
        "f_micro": partial(calc, func=sklearn.metrics.f1_score, average='micro'),
        "f_macro": partial(calc, func=sklearn.metrics.f1_score, average='macro'),
        "f_w": partial(calc, func=sklearn.metrics.f1_score, average='weighted'),
        "classificationReport": partial(calc, func=sklearn.metrics.classification_report, output_dict=True)
    }

    results, model_outputs, wrong_pred = model.evaluate(test,
                                                        verbose=True,
                                                        output_dir="/generated/other/catiss_reproduction.csv",
                                                        **metrics_recom)
    print(results)


def atomic_predict(model, dataset):
    # Predict every issue in the dataset individually
    predictions = []
    ground_truth = []
    sm_confidences = []
    pred_times = []
    supervisor_times = []
    for i, issue in tqdm(dataset.iterrows(), desc="predicting"):
        # track the time it takes to make a prediction
        start = time.time()
        _pred, _raw = model.predict([issue['text']])
        pred_times.append(time.time() - start)

        print(f"input{i}  pred={_pred} label={issue['labels']}")
        predictions.append(_pred[0])
        ground_truth.append(issue['labels'])

        # track the time it takes to compute and read softmax
        start = time.time()
        sm_confidences.append(scipy.special.softmax(_raw[0])[_pred[0]])
        supervisor_times.append(time.time() - start)
    return predictions, ground_truth, sm_confidences, pred_times, supervisor_times


def make_predictions():
    test = preprocess_inputs()

    test = test.head(NUM_SAMPLES)

    model = _load_model()
    predictions, gt_, sm_confidences, pred_times, supervisor_times = atomic_predict(model, test)

    # Micro-averaged F1 score
    print(f"Micro-averaged F1 score: {sklearn.metrics.f1_score(gt_, predictions, average='micro')}")

    assert test['labels'].tolist() == gt_, "Ground truth sanity check failed."

    res = pd.DataFrame({'index': list(range(len(predictions))),
                        'prediction': predictions,
                        'ground_truth': gt_,
                        'sm_confidence': sm_confidences,
                        'pred_time': pred_times,
                        'supervisor_time': supervisor_times})

    res.to_csv("/generated/predictions_and_uncertainties/issues_catiss.csv", index=False)


if __name__ == "__main__":
    make_predictions()
