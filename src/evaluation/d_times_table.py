import textwrap

import pandas as pd

from src.evaluation._data_loader import *


def compute_bisupervised_time(mean_local_time, mean_remote_time, remote_call_fraction):
    return mean_local_time + mean_remote_time * remote_call_fraction


def compute_time_break_even(mean_local_time, mean_remote_time):
    return 1 - (mean_local_time / mean_remote_time)


CASE_STUDY_REMARKS = {
    Imagenet().case_study_name: r"Remote model emulated locally.\\Both models use batched inference.",
    Imdb().case_study_name: r"",
    Issues().case_study_name: r"Remote model emulated locally.",
    SquadV2All().case_study_name: r"",
    SquadV2Possible().case_study_name: r"",
}

LINEBREAK = r'\\'

BREAK_EVEN_HEADER = r'\thead{Break Even\\(*)}'
LOCAL_AVG_HEADER = r'\thead{Local\\Only}'
REMOTE_AVG_HEADER = r'\thead{Remote\\Only}'
POINTS_HEADER = r'\thead{Eval Points\\(vs. Remote Only)}'
REMARKS_HEADER = 'Remarks'

if __name__ == '__main__':

    df = pd.DataFrame(
        columns=['Case Study', LOCAL_AVG_HEADER, REMOTE_AVG_HEADER, BREAK_EVEN_HEADER, POINTS_HEADER, REMARKS_HEADER])
    for i, cs in enumerate(CASE_STUDIES):
        break_even = compute_time_break_even(*cs.average_times)
        cp_lines = []
        for cp in cs.eval_points.keys():
            bisupervised_time = compute_bisupervised_time(*cs.average_times, cp)
            improvement = (cs.average_times[1] - bisupervised_time) / cs.average_times[1]
            cp_lines.append(f'{int(cp * 100)}\\%: {bisupervised_time:.2f}s (-{improvement * 100:.1f}\\%)')
        cp_lines = LINEBREAK.join(cp_lines)
        cp_lines = f"\\makecell{{{cp_lines}}}"

        vspace_ = r"\vspace{5pt}" if i != len(CASE_STUDIES) - 1 else ""
        df = df.append({
            'Case Study': f"\\makecell{{{cs.case_study_name.replace(' ', LINEBREAK, 1)}}}",
            LOCAL_AVG_HEADER: f'{cs.average_times[0]:.2f}s',
            REMOTE_AVG_HEADER: f'{cs.average_times[1]:.2f}s',
            BREAK_EVEN_HEADER: f'\\makecell{{{break_even * 100:.2f}\\%}}',
            POINTS_HEADER: cp_lines,
            REMARKS_HEADER: vspace_ + f"\\makecell[l]{{{CASE_STUDY_REMARKS[cs.case_study_name]}}}"
        }, ignore_index=True)

    with pd.option_context("max_colwidth", 1000):
        latex = df.to_latex(index=False, escape=False)

    pre = textwrap.dedent(r"""
    \begin{table}[ht]
    \footnotesize
    \centering
    """)

    latex = latex.replace(r'\bottomrule', r'\midrule').replace(r'\end{tabular}', textwrap.dedent(f"""
    \\multicolumn{{6}}{{l}}{{\\makecell{{(*) Break even point: the fraction of remote calls 
    at which the \\architecture latency equals the latency{LINEBREAK} of the remote-only approach.
    For any 1st level supervisor threshold leading to fewer remote predictions,{LINEBREAK}
    \\architecture is faster than using the standalone remote model. Hence, higher is better.}}}}{LINEBREAK}
    \\bottomrule
    \\end{{tabular}}
    """))

    post = textwrap.dedent(r"""
    \caption{Average Prediction and Supervision Latency per Input}
    \label{tab:times}
    \end{table}
    """)

    with open('/generated/results/table_times.tex', 'w') as f:
        f.write(pre + latex + post)

    print('Wrote times table.')
