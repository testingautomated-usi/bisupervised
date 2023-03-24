import os.path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.evaluation import _data_loader
from src.evaluation._data_loader import PredictionData


def plot_rac_curve(data: PredictionData) -> None:
    x_axis = list(data.rac_points().keys())
    y_axis_sysacc = [data.rac_points()[_id]["system_perf"] for _id in x_axis]

    plt.figure(figsize=(6, 4))

    plt.plot(x_axis, y_axis_sysacc, label="BiSupervised")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # rand_x = ["0%", "100%"]
    range_x = [0, 1]
    rand_y = [y_axis_sysacc[0], y_axis_sysacc[-1]]
    plt.plot(range_x, rand_y, "--", label="Random (expected value)")

    plt.annotate(
        f"Only remote\npredictions\n(100% / {rand_y[1]:.3f})",
        xy=(range_x[1], rand_y[1]), xytext=(-30, -70),
        textcoords='offset points', ha='center', va='bottom',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(
        f"Only local\npredictions\n(0% / {rand_y[0]:.3f})",
        xy=(range_x[0], rand_y[0]), xytext=(60, -10),
        textcoords='offset points', ha='center', va='bottom',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    for x, point_name in data.eval_points.items():
        # Get first point with x>=x (this will lead to x~= x, but not care about rounding errors)
        y = next(y for x_, y in data.rac_points().items() if x_ >= x)["system_perf"]
        plt.plot(x, y, 'ro')
        declaration = "remote-even\n" if point_name == "remote-even" else ""
        offset = -35 if point_name == "remote-even" else -20
        plt.annotate(
            f"{declaration}({int(round(x * 100))}% / {y:.3f})",
            xy=(x, y), xytext=(0, offset),
            textcoords='offset points', ha='center', va='bottom')

    plt.text(.01, .99, f"AUC-RAC = {data.auc_rac():.3f}", ha='left', va='top', transform=plt.gca().transAxes)

    plt.locator_params(axis='x', nbins=10)

    plt.legend(loc='lower right')
    plt.xlabel("Percentage of Samples predicted by Remote Model (=cost)")
    plt.ylabel(f"System-level {data.metric_name},\n without 2nd-level supervision")
    plt.tight_layout()

    if not os.path.exists("/generated/results/rac"):
        os.makedirs("/generated/results/rac")

    plt.savefig(f"/generated/results/rac/rac_{data.case_study_name}.svg")
    plt.savefig(f"/generated/results/rac/rac_{data.case_study_name}.png")
    # plt.show()
    plt.close()


if __name__ == '__main__':
    for case_study in _data_loader.CASE_STUDIES:
        plot_rac_curve(case_study)
