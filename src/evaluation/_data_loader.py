import abc
from functools import cached_property, lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

MAX_RAC_POINTS = 1000

PRED_DIR = "/generated/predictions_and_uncertainties"


def _get_eval_points_for_superaccurate_cs(rac_points: Dict[float, Dict[str, float]]) -> Dict[float, str]:
    # select point with the best system performance
    best_i = np.argmax([p["system_perf"] for p in rac_points.values()])
    best_x = list(rac_points.keys())[best_i]
    # select first point with at least the fully-remote performance
    remote_only = rac_points[1.0]["system_perf"]
    even_i = np.argmax([p["system_perf"] >= remote_only for p in rac_points.values()])
    even_x = list(rac_points.keys())[even_i]
    return {
        even_x: "remote-even",
        best_x: "best",
    }


def _s_beta(sys_perf: float, acceptance_rate: float, beta: float):
    """Compute the S_beta score, which is the harmonic mean between performance and acceptance rate.

    See Weiss and Tonella, STVR, 2023

    Note, as all our sys_perf scores are in [0,1], we can simplify the formula
    and do not need the normalization.
    """
    return (1 + beta ** 2) * sys_perf * acceptance_rate / (beta ** 2 * sys_perf + acceptance_rate)


class InfeasibleTargetFPRException(Exception):
    pass


class PredictionData(abc.ABC):

    @abc.abstractmethod
    def _load_local_correct_and_scores(self):
        """Load arrays representing local correctness and scores"""
        pass

    @abc.abstractmethod
    def _load_remote_correct_and_scores(self):
        """Load arrays representing remote correctness and scores"""
        pass

    @property
    @abc.abstractmethod
    def average_times(self) -> Tuple[float, float]:
        """Load arrays representing average times for local and remote blocks.

        This includes the times for prediction + supervision..."""
        pass

    @property
    @abc.abstractmethod
    def case_study_name(self):
        """Name of the case study"""
        pass

    @property
    @abc.abstractmethod
    def metric_name(self):
        """Name of the assessment metric used for this case study"""
        pass

    @cached_property
    def is_local_correct(self):
        local_correct, _ = self._load_local_correct_and_scores()
        return local_correct[:self._num_samples]

    @cached_property
    def is_remote_correct(self):
        remote_correct, _ = self._load_remote_correct_and_scores()
        return remote_correct[:self._num_samples]

    @cached_property
    def local_scores(self):
        _, local_scores = self._load_local_correct_and_scores()
        return local_scores[:self._num_samples]

    @cached_property
    def remote_scores(self):
        _, remote_scores = self._load_remote_correct_and_scores()
        return remote_scores[:self._num_samples]

    @property
    def eval_points(self) -> Dict[float, str]:
        """Points at which to evaluate the system performance.

        key = fraction of requests to send to remote,
        value = type of eval point ('best', 'remote-even', 'default')"""
        return {
            0.3: "default",
            0.5: "default",
            0.7: "default",
        }

    @property
    def target_fprs(self):
        return [0.01, 0.05, 0.1]

    @cached_property
    def _num_samples(self):
        local_correct, local_scores = self._load_local_correct_and_scores()
        remote_correct, remote_scores = self._load_remote_correct_and_scores()
        assert local_correct.shape == local_scores.shape
        assert remote_correct.shape == remote_scores.shape
        min_num_samples = min(local_correct.shape[0], remote_correct.shape[0])
        if local_correct.shape[0] != remote_correct.shape[0]:
            print(f"Warning: local and remote predictions have different lengths. "
                  f"Using the first {min_num_samples} samples.")
        return min_num_samples

    @staticmethod
    def _get_threshold(values: np.ndarray, cutoff: float):
        assert 0 <= cutoff <= 1, "Cutoff must be between 0 and 1"
        _values = np.copy(values)
        _values = np.sort(_values)
        num_cut = int(len(_values) * cutoff)
        if num_cut == 0:
            return _values[0]
        if num_cut == len(_values):
            return _values[-1]
        return _values[num_cut]

    def _acc(self, correct: np.ndarray) -> float:
        assert correct.ndim == 1, "Correct must be a 1D array"
        assert correct.dtype == bool or np.isin(correct, [0, 1]).all(), "Correct must be a boolean array"
        if len(correct) == 0:
            return np.nan
        return int(np.sum(correct)) / len(correct)

    @lru_cache
    def rac_points(self) -> Dict[float, Dict[str, float]]:
        """Points for the Request Accuracy Curve"""
        print(f"Computing RAC points for {self.case_study_name}")
        result = dict()
        num_steps_ = min(MAX_RAC_POINTS, self._num_samples)
        for cutoff in tqdm(np.arange(0.0, 1.0001, 1 / num_steps_),
                           desc=f"Calculating RAC points for {self.case_study_name}"):
            # cutoff = round(float(cutoff), 2)
            t = self._get_threshold(self.local_scores, cutoff)

            rejected_local = self.local_scores < t
            supervised_correct = np.copy(self.is_local_correct)
            supervised_correct[rejected_local] = self.is_remote_correct[rejected_local]
            result[cutoff] = {
                "system_perf": self._acc(supervised_correct),
                # "superv_acc": self._acc(self.is_local_correct[np.logical_not(rejected_local)]),
                # "healing_acc": self._acc(self.is_remote_correct[rejected_local]),
                # "acceptance_rate": int(np.sum(np.logical_not(rejected_local))) / len(self.is_local_correct),
            }
        return result

    def auc_rac(self):
        """Area under the Request Accuracy Curve"""
        print(f"Computing AUC for {self.case_study_name}")
        rac_points = self.rac_points()
        local_only = rac_points[0.0]["system_perf"]
        remote_only = rac_points[1.0]["system_perf"]
        avg_system_acc = np.mean([s["system_perf"] for s in rac_points.values()])
        return (avg_system_acc - local_only) / (remote_only - local_only)

    def models_complementarity_mean(self):
        """The percentage of inputs for which either local or remote are correct (but not both)."""
        return np.mean(np.logical_xor(self.is_local_correct, self.is_remote_correct))

    def both_correct_mean(self):
        """The percentage of inputs for which both local and remote are correct."""
        return np.mean(np.logical_and(self.is_local_correct, self.is_remote_correct))

    def both_incorrect_mean(self):
        """The percentage of inputs for which neither local nor remote are correct."""
        return np.mean(np.logical_not(np.logical_or(self.is_local_correct, self.is_remote_correct)))

    def only_local_correct_mean(self):
        """The percentage of inputs for which only local is correct."""
        return np.mean(np.logical_and(self.is_local_correct, np.logical_not(self.is_remote_correct)))

    def only_remote_correct_mean(self):
        """The percentage of inputs for which only remote is correct."""
        return np.mean(np.logical_and(np.logical_not(self.is_local_correct), self.is_remote_correct))

    def _supervised_performance(self, rejected_local, overall_rejected):
        # This is overridden in issues case study, to get weighted f1
        overall_correct = np.copy(self.is_local_correct)
        overall_correct[rejected_local] = self.is_remote_correct[rejected_local]
        overall_accepted_correct = overall_correct[np.logical_not(overall_rejected)]
        return self._acc(overall_accepted_correct)

    def _supervised_performance_mono(self, accepted, model: str):
        # This is overridden in issues case study, to get weighted f1
        if model == "local":
            correct = self.is_local_correct
        elif model == "remote":
            correct = self.is_remote_correct
        else:
            raise ValueError(f"Unknown model: {model}")
        return self._acc(correct[accepted])

    def bisupervised_supervision(self, local_cutoff: float, target_fpr: float):
        local_t = self._get_threshold(self.local_scores, local_cutoff)
        rejected_local = self.local_scores < local_t
        num_correct_local = np.sum(self.is_local_correct[np.logical_not(rejected_local)])

        remote_correct = self.is_remote_correct[rejected_local]
        remote_scores = self.remote_scores[rejected_local]

        # To get the target FPR, only the twice rejected, but correct remote predictions matter
        success = False
        sorted_remote_scores = np.sort(np.copy(remote_scores))
        for remote_threshold in tqdm(sorted_remote_scores):
            # FP are correct remote predictions that are rejected
            fp = np.sum(remote_correct[remote_scores < remote_threshold])
            # TN are all correctly accepted local predictions
            #   + all non-rejected correct remote predictions
            tn = num_correct_local + np.sum(remote_correct & (remote_scores >= remote_threshold))
            fpr = fp / (fp + tn)
            if fpr >= target_fpr:
                print(f"Found threshold {remote_threshold} for FPR {target_fpr}")
                success = True
                break

        if not success:
            print(f"Could not find threshold for FPR {target_fpr}")
            raise InfeasibleTargetFPRException()

        overall_rejected = np.copy(rejected_local)
        overall_rejected[rejected_local] = remote_scores < remote_threshold

        sys_acceptance_rate = int(np.sum(np.logical_not(overall_rejected))) / len(overall_rejected)
        sys_supervised_perf = self._supervised_performance(rejected_local, overall_rejected)
        return {
            "2nd_level_acceptance_rate": np.sum(remote_scores > remote_threshold) / len(remote_scores),
            "system_acceptance_rate": sys_acceptance_rate,
            "system_supervised_perf": sys_supervised_perf,
            "s_05": _s_beta(sys_supervised_perf, sys_acceptance_rate, 0.5),
            "s_1": _s_beta(sys_supervised_perf, sys_acceptance_rate, 1.0),
            "s_2": _s_beta(sys_supervised_perf, sys_acceptance_rate, 2.0),
        }

    def monosupervised_supervision(self, target_fpr, model: str):
        if model == "local":
            correct, scores = self.is_local_correct, self.local_scores
        elif model == "remote":
            correct, scores = self.is_remote_correct, self.remote_scores
        else:
            raise ValueError(f"Unknown model: {model}")

        # To get the target FPR, only the twice rejected, but correct remote predictions matter
        success = False
        for threshold in tqdm(np.sort(np.copy(scores))):
            # FP are correct predictions that are rejected
            fp = np.sum(correct[scores < threshold])
            # TN are all correctly accepted predictions
            tn = np.sum(correct[scores >= threshold])
            fpr = fp / (fp + tn)
            if fpr >= target_fpr:
                print(f"Found monosupervised threshold {threshold} for FPR {target_fpr}")
                success = True
                break

        if not success:
            raise ValueError(f"Could not find monosupervised threshold for FPR {target_fpr}."
                             f"This cannot happen.")

        acceptance_rate = np.sum(scores > threshold) / len(scores)
        supervised_perf = self._supervised_performance_mono(accepted=scores > threshold,
                                                            model=model)

        return {
            "2nd_level_acceptance_rate": "N/A",
            "system_acceptance_rate": acceptance_rate,
            "system_supervised_perf": supervised_perf,
            "s_05": _s_beta(supervised_perf, acceptance_rate, 0.5),
            "s_1": _s_beta(supervised_perf, acceptance_rate, 1.0),
            "s_2": _s_beta(supervised_perf, acceptance_rate, 2.0),
        }


class Imagenet(PredictionData):

    @property
    @lru_cache
    def average_times(self) -> Tuple[float, float]:
        local_df = pd.read_csv(f"{PRED_DIR}/imagenet_local.csv")
        local_times = local_df["pred_time"] + local_df["supervisor_time"]
        remote_df = pd.read_csv(f"{PRED_DIR}/imagenet_remote.csv")
        remote_times = remote_df["pred_time"] + remote_df["supervisor_time"]
        return float(np.mean(local_times)), float(np.mean(remote_times))

    @property
    def case_study_name(self) -> str:
        return "ImageNet"

    @property
    def metric_name(self):
        return "top-1 accuracy"

    @lru_cache
    def _load_local_correct_and_scores(self) -> (np.ndarray, np.ndarray):
        df = pd.read_csv(f"{PRED_DIR}/imagenet_local.csv")
        predictions = df["prediction"].to_numpy()
        labels = df["ground_truth"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        correct = predictions == labels
        return correct, scores

    @lru_cache
    def _load_remote_correct_and_scores(self) -> (np.ndarray, np.ndarray):
        df = pd.read_csv(f"{PRED_DIR}/imagenet_remote.csv")
        predictions = df["prediction"].to_numpy()
        labels = df["ground_truth"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        correct = predictions == labels
        return correct, scores


class SquadV2Possible(PredictionData):

    @property
    def metric_name(self):
        return "exact match (EM)"

    @lru_cache
    def _load_local_correct_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/squadv2_local.csv")
        exact_match = df["exact_match"].to_numpy()
        possible = df["is_possible"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        return exact_match[possible], scores[possible]

    def _load_remote_correct_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/squadv2_remote.csv")
        exact_match = df["exact_match"].to_numpy()
        possible = df["is_possible"].to_numpy()
        scores = df["gpt3_confidence"].to_numpy()
        return exact_match[possible], scores[possible]

    @property
    def case_study_name(self):
        return "SQuADv2 (possible only)"

    @property
    def eval_points(self) -> Dict[float, str]:
        return _get_eval_points_for_superaccurate_cs(self.rac_points())

    @property
    @lru_cache
    def average_times(self) -> Tuple[float, float]:
        df_local = pd.read_csv(f"{PRED_DIR}/squadv2_local.csv")
        possible = df_local["is_possible"].to_numpy()
        df_remote = pd.read_csv(f"{PRED_DIR}/squadv2_remote.csv")

        local_times = df_local["pred_time"].to_numpy()
        remote_times = df_remote["pred_time"].to_numpy()

        # Only consider possible question
        local_times = local_times[possible]
        remote_times = remote_times[possible]

        return float(np.mean(local_times)), float(np.mean(remote_times))


class SquadV2All(PredictionData):

    @property
    def metric_name(self):
        return "exact match (EM)"

    @lru_cache
    def _load_local_correct_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/squadv2_local.csv")
        exact_match = df["exact_match"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        return exact_match, scores

    def _load_remote_correct_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/squadv2_remote.csv")
        exact_match = df["exact_match"].to_numpy()
        scores = df["gpt3_confidence"].to_numpy()
        return exact_match, scores

    @property
    def case_study_name(self):
        return "SQuADv2 (all)"

    @property
    def eval_points(self) -> Dict[float, str]:
        return _get_eval_points_for_superaccurate_cs(self.rac_points())

    @property
    @lru_cache
    def average_times(self) -> Tuple[float, float]:
        df_local = pd.read_csv(f"{PRED_DIR}/squadv2_local.csv")
        df_remote = pd.read_csv(f"{PRED_DIR}/squadv2_remote.csv")
        return float(np.mean(df_local["pred_time"])), float(np.mean(df_remote["pred_time"]))


class Issues(PredictionData):

    @property
    def metric_name(self):
        return "macro F1"

    @lru_cache
    def _load_local_pred_gt_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/issues_local.csv")
        predictions = df["prediction"].to_numpy()
        labels = df["ground_truth"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        return predictions, labels, scores

    @lru_cache
    def _load_remote_pred_gt_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/issues_catiss.csv")
        predictions = df["prediction"].to_numpy()
        labels = df["ground_truth"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        return predictions, labels, scores

    @property
    @lru_cache
    def average_times(self) -> Tuple[float, float]:
        df_local = pd.read_csv(f"{PRED_DIR}/issues_local.csv")
        local_times = df_local["pred_time"] + df_local["supervisor_time"]
        df_remote = pd.read_csv(f"{PRED_DIR}/issues_catiss.csv")
        remote_times = df_remote["pred_time"] + df_remote["supervisor_time"]
        return float(np.mean(local_times)), float(np.mean(remote_times))

    def _load_local_correct_and_scores(self):
        predictions, labels, scores = self._load_local_pred_gt_and_scores()
        correct = predictions == labels
        return correct, scores

    def _load_remote_correct_and_scores(self):
        predictions, labels, scores = self._load_remote_pred_gt_and_scores()
        correct = predictions == labels
        return correct, scores

    @property
    def _ground_truth(self):
        _, gt, _ = self._load_local_pred_gt_and_scores()
        return gt[:self._num_samples]

    @property
    def local_pred(self):
        pred, _, _ = self._load_local_pred_gt_and_scores()
        return pred[:self._num_samples]

    @property
    def remote_pred(self):
        pred, _, _ = self._load_remote_pred_gt_and_scores()
        return pred[:self._num_samples]

    @staticmethod
    def _macro_f1(y_true, y_pred) -> float:
        return f1_score(y_true, y_pred, average="macro")

    @lru_cache
    def rac_points(self) -> Dict[float, Dict[str, float]]:
        """Points for the Request Accuracy Curve using micro F1"""
        print(f"Computing RAC points for {self.case_study_name}")
        result = dict()
        num_steps_ = min(MAX_RAC_POINTS, self._num_samples)
        for cutoff in tqdm(np.arange(0.0, 1.0001, 1 / num_steps_),
                           desc=f"Calculating RAC points for {self.case_study_name}"):
            # cutoff = round(float(cutoff), 2)
            t = self._get_threshold(self.local_scores, cutoff)

            rejected_local = self.local_scores < t
            supervised_pred = np.copy(self.local_pred)
            supervised_pred[rejected_local] = self.remote_pred[rejected_local]
            result[cutoff] = {
                "system_perf": self._macro_f1(supervised_pred, self._ground_truth),
                # "superv_acc": self._acc(self.is_local_correct[np.logical_not(rejected_local)]),
                # "healing_acc": self._acc(self.is_remote_correct[rejected_local]),
                # "acceptance_rate": int(np.sum(np.logical_not(rejected_local))) / len(self.is_local_correct),
            }
        return result

    @property
    def case_study_name(self):
        return "Issues"

    def _supervised_performance(self, rejected_local, overall_rejected):
        # This is overridden in issues case study, to get weighted f1
        overall_pred = np.copy(self.local_pred)
        overall_pred[rejected_local] = self.remote_pred[rejected_local]
        overall_pred_accepted = overall_pred[np.logical_not(overall_rejected)]
        overall_gt_accepted = self._ground_truth[np.logical_not(overall_rejected)]
        return self._macro_f1(overall_pred_accepted, overall_gt_accepted)

    def _supervised_performance_mono(self, accepted, model: str):
        # This is overridden in issues case study, to get weighted f1
        selected_gt = self._ground_truth[accepted]
        if model == "local":
            selected_pred = self.local_pred[accepted]
        elif model == "remote":
            selected_pred = self.remote_pred[accepted]
        else:
            raise ValueError(f"Unknown model {model}")
        return self._macro_f1(selected_gt, selected_pred)


class Imdb(PredictionData):

    @property
    def average_times(self) -> Tuple[float, float]:
        df_local = pd.read_csv(f"{PRED_DIR}/imdb_local.csv")
        local_times = df_local["pred_time"] + df_local["supervisor_time"]
        df_remote = pd.read_csv(f"{PRED_DIR}/imdb_remote.csv")
        remote_times = df_remote["pred_time"] + df_remote["supervisor_time"]
        return float(np.mean(local_times)), float(np.mean(remote_times))

    @property
    def metric_name(self):
        return "top-1 accuracy"

    def _load_local_correct_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/imdb_local.csv")
        predictions = df["prediction"].to_numpy()
        labels = df["ground_truth"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        return predictions == labels, scores

    def _load_remote_correct_and_scores(self):
        df = pd.read_csv(f"{PRED_DIR}/imdb_remote.csv")
        predictions = df["prediction"].to_numpy()
        labels = df["ground_truth"].to_numpy()
        scores = df["sm_confidence"].to_numpy()
        return predictions == labels, scores

    @property
    def case_study_name(self):
        return "imdb"

    @property
    def eval_points(self) -> Dict[float, str]:
        return _get_eval_points_for_superaccurate_cs(self.rac_points())


CASE_STUDIES = [
    Imdb(),
    Issues(),
    Imagenet(),
    SquadV2Possible(),
    SquadV2All(),
]
