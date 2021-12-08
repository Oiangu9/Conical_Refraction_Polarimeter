import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(f"../../..")


def plot_histograms_for(variable, final_results_df, experiment_name):
    categories=len(final_results_df.groupby([variable]))
    columns=len(final_results_df.groupby(['sigma_WN']))
    rows=1+len(final_results_df.groupby(['relative_saturation']))
    fig=plt.figure(figsize=(7*columns, 5*rows))
    main_h = fig.add_subplot(rows,2, 2)
    main_h=plt.subplot2grid((rows, columns), (0, 1), rowspan=1, colspan=2)
    #i=0
    bins_main_exponents=[1,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,0]
    bins_main=10**np.array(bins_main_exponents)
    bins_main[0]=0
    bins=np.array([0,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 ])
    #bin_widths=bins[1:]-bins[:-1]
    ax_title=fig.add_subplot(rows, columns, 1)
    ax_title.text(0, 0.3, f" Histograms for {variable}: \n\n\n The first plot in the right makes no groups\n other than {variable}, it is all the avilable data.\n\n\n The x axes represent the smallest\n absolute difference between the\n theoretical angle difference and\n the found angle difference,\n among the employed algorithms.", style='italic', fontsize=13,bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 10})
    ax_title.set_axis_off()
    for group_var_val, group_df in final_results_df.groupby([variable], sort=True):
        main_h.hist(group_df['min_abs_theoretical_error'], bins=bins_main, range=(0,0.4), label=f"{variable}={group_var_val}", rwidth=1, align='mid', edgecolor="k", alpha=0.6)
        i=columns+1
        for satur, satur_df in group_df.groupby(['relative_saturation'], sort=True):
            for sigma, sigma_satur_df in satur_df.groupby(['sigma_WN'], sort=True):
                ax=fig.add_subplot(rows,columns, i)
                ax.hist(sigma_satur_df['min_abs_theoretical_error'], bins=bins, range=(0,0.4), label=f"{variable}={group_var_val}", rwidth=1, align='mid', edgecolor="k", alpha=0.6)
                ax.set_title(f'sigma_WN={sigma} satur={satur}')
                ax.set_xscale('log')
                ax.legend()
                ax.grid(True)
                i+=1
    main_h.set_xscale('log')
    main_h.legend()
    main_h.grid(True)
    main_h.set_title(f"Experiment: {experiment_name}")
    #fig.suptitle(f"Histogrms for {variable} \n\n Experiment: {experiment_name}\n\n\n The x axes represent the smallest absolute difference between the theoretical\n angle difference and the found angle difference, among the employed algorithms")

    os.makedirs(f"./OUTPUT/PIPELINE/{experiment_name}/HISTOGRAMS/", exist_ok=True)
    #fig.tight_layout()
    plt.savefig(f"./OUTPUT/PIPELINE/{experiment_name}/HISTOGRAMS/Histogram_for_{variable}.png", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    for experiment_name in ["nx540_no_interpolation_in_iX_yes_recentering_average_nx_540_iX_378_angles_rad", "nx540_yes_interpolation_in_iX_no_recentering_average_nx_540_iX_378_angles_rad", "nx540_no_interpolation_in_iX_no_recentering_average_nx_540_iX_378_angles_rad", "nx1620_no_interpolation_in_iX_yes_recentering_average_nx_1620_iX_1134_angles_rad", "nx1620_no_interpolation_in_iX_no_recentering_average_nx_1620_iX_1134_angles_rad", "nx1620_yes_interpolation_in_iX_no_recentering_average_nx_1620_iX_1134_angles_rad"]:

        final_results_dict = json.load(open(f"./OUTPUT/PIPELINE/{experiment_name}/FINAL_RESULTS_{experiment_name}.json"))
        final_results_df = pd.DataFrame.from_dict(final_results_dict)
        final_results_df=final_results_df.astype({"R0":float, "w0":float, "sigma_WN":float, "relative_saturation":float})

        plot_histograms_for('R0', final_results_df, experiment_name)
        plot_histograms_for('w0', final_results_df, experiment_name)
        plot_histograms_for('interpolation', final_results_df, experiment_name)
        plot_histograms_for('averaged_images_or_angles', final_results_df, experiment_name)
        plot_histograms_for('best_algorithm', final_results_df, experiment_name)
