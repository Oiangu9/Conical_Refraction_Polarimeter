import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from styleframe import StyleFrame


def compute_expectation_CI(empirical_pdf, boots_samples, confidence):
    resamplings=np.random.choice(empirical_pdf, size=( boots_samples, empirical_pdf.shape[0]))
    boot_means=np.mean(resamplings, axis=1)
    boot_stds=np.std(resamplings, axis=1)
    observed_mean=empirical_pdf.mean()
    observed_std=empirical_pdf.std()
    boots_t=(observed_mean-boot_means)*np.sqrt(empirical_pdf.shape[0])/boot_stds
    boots_t_percentiles = np.percentile(boots_t, q=((100-confidence)/2, confidence+(100-confidence)/2))
    return observed_mean+boots_t_percentiles*observed_std/np.sqrt(empirical_pdf.shape[0])

def angle_to_pi_pi(angles): # convert any angle to range (-pi,pi]
    angles= angles%(2*np.pi) # take it to [-2pi, 2pi]
    return np.where( np.abs(angles)>np.pi, angles-np.sign(angles)*2*np.pi, angles)

def plot_histograms_for(variable, final_results_df, experiment_name, best_cis, conf):
    categories=len(final_results_df.groupby([variable]))
    columns=1 if categories==1 else 2 if (categories==2 or categories==4) else 3
    rows=categories//3+(categories%3!=0)

    fig=plt.figure(figsize=(7*columns, 5*rows))
    i=1
    bins_main_exponents=[1,-8.5, -8, -7.5, -7, -6.5,-6,-5.5,-5,-4.5,-4,-3.5,-1,0]
    bins_main=10**np.array(bins_main_exponents)
    bins_main[0]=0
    axs=[]
    maxy=0
    for group_var_val, group_df in final_results_df.groupby([variable], sort=True):
        axs.append(fig.add_subplot(rows,columns, i))
        axs[-1].hist(group_df['min_abs_theoretical_error'], bins=bins_main, range=(0,0.4), label=f"{variable}={group_var_val}", rwidth=1, align='mid', edgecolor="k", alpha=0.6)
        axs[-1].set_xscale('log')
        axs[]-1].legend()
        axs[-1].grid(True)
        axs[-1].set_title(f"Best {conf}% CI: {best_cis[i-1]}")
        i+=1
        maxy= if >maxy else maxy
    for ax in axs:
        ax.set_ylim(0,maxy)
    #fig.supylabel('common_y')
    #fig.suptitle(f"Histogrms for {variable} \n\n Experiment: {experiment_name}\n\n\n The x axes represent the smallest absolute difference between the theoretical\n angle difference and the found angle difference, among the employed algorithms")

    os.makedirs(f"./OUTPUT/PIPELINE/{experiment_name}/HISTOGRAMS/", exist_ok=True)
    #fig.tight_layout()
    plt.savefig(f"./OUTPUT/PIPELINE/{experiment_name}/HISTOGRAMS/Histogram_for_{variable}.png", bbox_inches='tight')
    plt.close()
    # FALTA ACUMULAR EL MAXIMO DE Y Y HACER AL FINAL EL LOOP PARA SETEAR ESE MAXIMO

def compute_bootstrap_t_mean_CIs_by(grouping_variables, final_results_df, experiment_name, boots_sampless, confidence=95):
    writer = StyleFrame.ExcelWriter(f"./OUTPUT/PIPELINE/{experiment_name}/EXCEL_{confidence}_CIs_{experiment_name}.xlsx")
    StyleFrame.A_FACTOR=10
    StyleFrame.P_FACTOR=0.9

    for grouping_variable, boots_samples in zip(grouping_variables, boots_sampless):
        ci_s = {f"Grouping Variable":[], "Group Value":[], "Best Results obs mean":[], f"Best results {confidence}% CI":[], "mirror_fibo CI":[], "mirror_quad CI":[], "rotation_fibo CI":[], "rotation_quad CI":[]}
        for var, var_df in final_results_df.groupby([grouping_variable]):
            ci_s["Grouping Variable"].append(grouping_variable)
            ci_s["Group Value"].append(var)
            # Perform a bootstrap test for each group for the mean. Both taking the best results
            # and grouping by algorithm
            # First of all for the mixed case (taking the best algorithms):
            empirical_pdf = var_df["min_abs_theoretical_error"].to_numpy()
            ci_s["Best Results obs mean"].append(empirical_pdf.mean())
            ci_s[f"Best results {confidence}% CI"].append(compute_expectation_CI(empirical_pdf, boots_samples=boots_samples, confidence=confidence))
            print(f"Done! for {grouping_variable} {var}")
            # And now we shall do it by algorithm
            ground_truth_relative_angles=angle_to_pi_pi((var_df["th_phiCR_prob"]-var_df["th_phiCR_ref"]).to_numpy())
            for alg in ["mirror_fibo", "rotation_fibo", "mirror_quad", "rotation_quad"]:
                ci_s[f"{alg} CI"].append(compute_expectation_CI(np.abs( var_df[alg]-ground_truth_relative_angles ),boots_samples=boots_samples, confidence=confidence  ) )

        ci_df = pd.DataFrame.from_dict(ci_s)

        # Convert the dataframe to an XlsxWriter Excel object.

        StyleFrame(ci_df).set_row_height(1,50).to_excel(writer, best_fit=list(ci_df.columns), sheet_name=f'CIs per {grouping_variable}, {boots_samples} resamplings', index=False,  float_format="%.8f")

        plot_histograms_for(grouping_variable, final_results_df, experiment_name, best_cis=ci_s[f"Best results {confidence}% CI"], conf=confidence)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()







if __name__ == '__main__':
    randomization_seed=666
    np.random.seed(randomization_seed)

    for experiment_name in ["fix_R0_vary_w0__nx_540_iX_378_angles_rad"]:# ["nx540_no_interpolation_in_iX_yes_recentering_average_nx_540_iX_378_angles_rad", "nx540_yes_interpolation_in_iX_no_recentering_average_nx_540_iX_378_angles_rad", "nx540_no_interpolation_in_iX_no_recentering_average_nx_540_iX_378_angles_rad", "nx1620_no_interpolation_in_iX_yes_recentering_average_nx_1620_iX_1134_angles_rad", "nx1620_no_interpolation_in_iX_no_recentering_average_nx_1620_iX_1134_angles_rad", "nx1620_yes_interpolation_in_iX_no_recentering_average_nx_1620_iX_1134_angles_rad"]:

        final_results_dict = json.load(open(f"./OUTPUT/PIPELINE/{experiment_name}/FINAL_RESULTS_{experiment_name}.json"))
        final_results_df = pd.DataFrame.from_dict(final_results_dict)
        final_results_df=final_results_df.astype({"R0":float, "w0":float})


        compute_bootstrap_t_mean_CIs_by(['rho0', 'w0','R0'], final_results_df, experiment_name, boots_sampless=[10000, 10000, 2000])
