import pandas as pd

from src.evaluation._data_loader import CASE_STUDIES

if __name__ == '__main__':

    res = []
    for cs in CASE_STUDIES:
        rac_points = cs.rac_points()
        local_only_perf = rac_points[0]
        remote_only_perf = rac_points[1]
        res.append(f"{cs.case_study_name} values. "
              f"Local only: {round(local_only_perf['system_perf'], 5)}, "
              f"Remote only {round(remote_only_perf['system_perf'], 5)} ")
              
    for r in res:
    	print(r)
