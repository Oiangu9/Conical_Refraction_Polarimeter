import sys
from SOURCE.GPU_Classes import *
import numpy as np


if __name__ == '__main__':
    # PARAMETER SETTINGS ##############################################
    ###################################################################
    ##################################################################
    # 0. GENERAL SETTINGS #############################################
    ################################################################
    experiment_name="BEST_RHO_0"
    randomization_seed=666

    # 1. SIMULATION ####################################################
    # Define the PARAMETERS #########################################
    # Ring parameters to test (each will be a different simulation)
    reference_theoretical_phiCR=[]
    reference_rho0_s=[]
    reference_w0_s=[]

    problem_theoretical_phiCRs=[]
    problem_rho0_s=[]
    problem_w0_s=[]

    resolution_side=450 # generated images will be resolution_side x resolution_side
    # Other parameters
    max_k=50
    num_k=1200
    sim_chunk_ax=550

    # 2. WHITE NOISE ####################################################
    # Define the PARAMETERS #########################################
    number_of_samples_per_sigma_and_reference=1 # they will be simulated once but different noises will be inserted
    number_of_samples_per_sigma_and_problem=2

    noise_sigmas=[] # same white noise for reference and problem linked pairs will be used
    #noise_sigmas_reference=[] # the standard deviation of the gaussian noise in each pixel
    #noise_sigmas_problem=[] # in principle the noise in the reference and the problem should be the same

    # 3. SATURATION ####################################################
    saturations_at_relative_intesities=[] # capping will be performed at different relative intensities
    # the profile will be multiplied by a factor and then capped (not capped directlty, since that
    # would not replicate the experimental capping)

    # 4. GRAVICENTER iX #################################################
    X=resolution_side*1.2/2

    # 5. POLARIZATION RELATIVE ANGLES ###################################
    # Mirror with affine interpolation & Rotation Algorithms will be employed
    # Each using both Fibonacci and Quadratic Fit Search
    # Results will be gathered in a table and outputed as an excel csv


    # 6. OUTPUT RESULTS INTO A LATEX? INTO AN EXCEL WITH IMAGES (GIFS) WOULD BE FANTASTIC


    # PARAMETER SETTINGS ##############################################
    ###################################################################
    ##################################################################
    # 0. GENERAL SETTINGS #############################################
    ################################################################
    os.makedirs(f'./PIPELINE/{experiment_name}/SIMULATIONS/', exist_ok=True)
    os.makedirs(f'./PIPELINE/{experiment_name}/SIMULATIONS/', exist_ok=True)
    os.makedirs(f'./PIPELINE/{experiment_name}/SIMULATIONS/', exist_ok=True)
    os.makedirs(f'./PIPELINE/{experiment_name}/SIMULATIONS/', exist_ok=True)

    np.random.seed(randomization_seed)


    # 1. SIMULATION ####################################################
    # Define the PARAMETERS #########################################
    # Ring parameters to test (each will be a different simulation)
    reference_theoretical_phiCR=[]
    reference_rho0_s=[]
    reference_w0_s=[]

    problem_theoretical_phiCRs=[]
    problem_rho0_s=[]
    problem_w0_s=[]

    resolution_side=450 # generated images will be resolution_side x resolution_side
    # Other parameters
    max_k=50
    num_k=1200
    sim_chunk_ax=550

    # 2. WHITE NOISE ####################################################
    # Define the PARAMETERS #########################################
    number_of_samples_per_sigma_and_reference=1 # they will be simulated once but different noises will be inserted
    number_of_samples_per_sigma_and_problem=2

    noise_sigmas=[] # same white noise for reference and problem linked pairs will be used
    #noise_sigmas_reference=[] # the standard deviation of the gaussian noise in each pixel
    #noise_sigmas_problem=[] # in principle the noise in the reference and the problem should be the same

    # 3. SATURATION ####################################################
    saturations_at_relative_intesities=[] # capping will be performed at different relative intensities
    # the profile will be multiplied by a factor and then capped (not capped directlty, since that
    # would not replicate the experimental capping)

    # 4. GRAVICENTER iX #################################################
    X=resolution_side*1.2/2

    # 5. POLARIZATION RELATIVE ANGLES ###################################
    # Mirror with affine interpolation & Rotation Algorithms will be employed
    # Each using both Fibonacci and Quadratic Fit Search
    # Results will be gathered in a table and outputed as an excel csv


    # 6. OUTPUT RESULTS INTO A LATEX? INTO AN EXCEL WITH IMAGES (GIFS) WOULD BE FANTASTIC
    # Instatiate simulator and run simulations
    simulator=RingSimulator_Optimizer_GPU( n=1.5, w0=1, a0=1.0, max_k=max_k, num_k=num_k, nx=resolution_side, sim_chunk_x=sim_chunk_ax, sim_chunk_y=sim_chunk_ax)
