import os
from typing import List

import pandas as pd
from pandas import MultiIndex

from src.evaluation._data_loader import PredictionData, InfeasibleTargetFPRException, CASE_STUDIES

KEY_LOCAL_BASELINE = "0.00 (local_baseline)"
KEY_REMOTE_BASELINE = "1.00 (remote_baseline)"


def _ep_key(eval_point):
    return f"{eval_point[0]:.2f} ({eval_point[1]})"


def build_empty_table(case_studies: List[PredictionData]):
    rows = []
    for case_study in case_studies:
        for target_fpr in case_study.target_fprs:
            rows.append((case_study.case_study_name, target_fpr, KEY_LOCAL_BASELINE))
            for eval_point in case_study.eval_points.items():
                rows.append((case_study.case_study_name, target_fpr, _ep_key(eval_point)))
            # Works, but was not part of the registered report
            rows.append((case_study.case_study_name, target_fpr, KEY_REMOTE_BASELINE))

    frame = pd.DataFrame(
        columns=["remote_delta", "sys_delta", "sys_obj", "sys_s05", "sys_s1", "sys_s2"],
        index=MultiIndex.from_tuples(rows, names=["case_s", "target_fpr", "1st_reject_rate"])
    )

    frame.sort_index()

    return frame


def fill_table_for_case_study(case_study: PredictionData, tab: pd.DataFrame):
    cs_name = case_study.case_study_name
    for target_fpr in case_study.target_fprs:
        # Adding this now saves us from sorting later

        for ep in case_study.eval_points.items():
            try:
                bisup_res = case_study.bisupervised_supervision(local_cutoff=ep[0], target_fpr=target_fpr)
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "remote_delta"] = bisup_res["2nd_level_acceptance_rate"]
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "sys_delta"] = bisup_res["system_acceptance_rate"]
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "sys_obj"] = bisup_res["system_supervised_perf"]
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "sys_s05"] = bisup_res["s_05"]
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "sys_s1"] = bisup_res["s_1"]
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "sys_s2"] = bisup_res["s_2"]
            except InfeasibleTargetFPRException:
                tab.loc[(cs_name, target_fpr, _ep_key(ep)), "remote_delta"] = "infeasible"
        for model, baseline_name in (
                ("local", KEY_LOCAL_BASELINE),
                ("remote", KEY_REMOTE_BASELINE)  # Works, but was not part of the registered report
        ):
            sup_res = case_study.monosupervised_supervision(target_fpr, model)
            tab.loc[(cs_name, target_fpr, baseline_name), "sys_delta"] = sup_res["system_acceptance_rate"]
            tab.loc[(cs_name, target_fpr, baseline_name), "sys_obj"] = sup_res["system_supervised_perf"]
            tab.loc[(cs_name, target_fpr, baseline_name), "sys_s05"] = sup_res["s_05"]
            tab.loc[(cs_name, target_fpr, baseline_name), "sys_s1"] = sup_res["s_1"]
            tab.loc[(cs_name, target_fpr, baseline_name), "sys_s2"] = sup_res["s_2"]


def table_to_latex(t: pd.DataFrame, case_study: PredictionData):
    t = t.round(2)
    t = t.fillna("n/a")

    # Drop all rows that are not part of the case study
    t = t.loc[t.index.get_level_values("case_s") == case_study.case_study_name]

    # Dont include remote-only baselines
    t = t.loc[t.index.get_level_values("1st_reject_rate") != KEY_REMOTE_BASELINE]

    # Remove the case study column
    t = t.droplevel("case_s")

    ls = t.to_latex()

    # Replace 1st level key names
    ls = ls.replace("local\_baseline", "baseline"). \
        replace("remote\_baseline", "remote-only").replace("(default)", "")

    # Replace header names
    ls = ls.replace("case\_s", "study")
    ls = ls.replace("target\_fpr", "FPR")
    ls = ls.replace("1st\_reject\_rate", "remote requests")
    ls = ls.replace("remote\_delta", r"remote $\Delta$")
    ls = ls.replace("sys\_delta", r"$\Delta$")
    ls = ls.replace("sys\_obj", r"$\overline{obj}$")
    ls = ls.replace("sys\_s05", r"$S_{0.5}$")
    ls = ls.replace("sys\_s1", r"$S_1$")
    ls = ls.replace("sys\_s2", r"$S_2$")

    # ls = ls.replace("imdb", r"\multirow{15}{*}{\verti{Imdb}}")
    # ls = ls.replace("Issues", r"\multirow{15}{*}{\verti{Issues}}")
    # ls = ls.replace("ImageNet", r"\multirow{15}{*}{\verti{ImageNet}}")
    # ls = ls.replace("SQuADv2 (possible only)", r"\multirow{12}{*}{\verti{SQuADv2\\(possible only)}}")
    # ls = ls.replace("SQuADv2 (all)", r"\multirow{12}{*}{\verti{SQuADv2\\(all)}}")

    num_rows_per_fpr = len(case_study.eval_points) + 1
    ls = ls.replace("0.01", r"\multirow{" + str(num_rows_per_fpr) + r"}{*}{\verti{0.01}}")
    ls = ls.replace("0.05", r"\midrule \multirow{" + str(num_rows_per_fpr) + r"}{*}{\verti{0.05}}")
    ls = ls.replace("0.10", r"\midrule \multirow{" + str(num_rows_per_fpr) + r"}{*}{\verti{0.10}}")


    ls = ls.replace("lllrrrrr", "llcccccc")

    latex_pretext = r"""
    \begin{table}
    \newcommand{\verti}[1]{\begin{tabular}{@{}c@{}}\rotatebox[origin=c]{90}{\parbox{1cm}{\centering #1}}\end{tabular}}
    """

    latex_posttext = f"""
    \\caption{{System-Level Supervised Assessment for {case_study.case_study_name}}}
    \\label{{tab:rq2_res_{case_study.case_study_name}}} 
    \\end{{table}}
    """
    return latex_pretext + ls + latex_posttext


if __name__ == '__main__':
    table = build_empty_table(CASE_STUDIES)
    for case_study in CASE_STUDIES:
        fill_table_for_case_study(case_study, table)

    table.to_csv("/generated/results/rq2_table.csv")
    #

    # ========
    # Create latex tables
    # ========
    table = pd.read_csv("/generated/results/rq2_table.csv", index_col=[0, 1, 2])
    tex_folder = "/generated/results/rq2_tex_tables/"
    if not os.path.exists(tex_folder):
        os.makedirs(tex_folder)

    for case_study in [cs for cs in CASE_STUDIES]:
        ltex = table_to_latex(table, case_study)
        with open(f"{tex_folder}rq2_table_{case_study.case_study_name}.tex", "w") as f:
            f.write(ltex)



