import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_bootstrap_t_mean_CIs_by(grouping_variable, final_results_df, experiment_name):


    os.makedirs(f"./OUTPUT/PIPELINE/{experiment_name}/CI_s/", exist_ok=True)


if __name__ == '__main__':

    for experiment_name in ["nx540_no_interpolation_in_iX_yes_recentering_average_nx_540_iX_378_angles_rad", "nx540_yes_interpolation_in_iX_no_recentering_average_nx_540_iX_378_angles_rad", "nx540_no_interpolation_in_iX_no_recentering_average_nx_540_iX_378_angles_rad", "nx1620_no_interpolation_in_iX_yes_recentering_average_nx_1620_iX_1134_angles_rad", "nx1620_no_interpolation_in_iX_no_recentering_average_nx_1620_iX_1134_angles_rad", "nx1620_yes_interpolation_in_iX_no_recentering_average_nx_1620_iX_1134_angles_rad"]:

        final_results_dict = json.load(open(f"./OUTPUT/PIPELINE/{experiment_name}/FINAL_RESULTS_{experiment_name}.json"))
        final_results_df = pd.DataFrame.from_dict(final_results_dict)
        final_results_df=final_results_df.astype({"R0":float, "w0":float, "sigma_WN":float, "relative_saturation":float})
        print(final_results_df.dtypes)


        compute_bootstrap_t_mean_CIs_by('R0', final_results_df, experiment_name)
        compute_bootstrap_t_mean_CIs_by('w0', final_results_df, experiment_name)
        compute_bootstrap_t_mean_CIs_by('interpolation', final_results_df, experiment_name)
        compute_bootstrap_t_mean_CIs_by('averaged_images_or_angles', final_results_df, experiment_name)
        compute_bootstrap_t_mean_CIs_by('best_algorithm', final_results_df, experiment_name)
