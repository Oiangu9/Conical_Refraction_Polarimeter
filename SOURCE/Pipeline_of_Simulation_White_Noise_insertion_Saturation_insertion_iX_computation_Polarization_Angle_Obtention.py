import sys
from SOURCE.GPU_Classes import *
import numpy as np


if __name__ == '__main__':
    # GENERAL SETTINGS #############################################
    ################################################################
    experiment_name="BEST_RHO_0"
    np.random.seed(666)
    reference_theoretical_phiCR=[]
    problem_theoretical_phiCRs=[]
    reference_rho0_s=[]
    problem_rho0_s=[]
    number_of_samples_per_reference=1 # they will be simulated once but different noises will be inserted
    number_of_samples_per_problem=2
    noise_sigmas_reference=[] # the standard deviation of the gaussian noise in each pixel
    noise_sigmas_problem=[] # in principle the noise in the reference and the problem should be the same





    # SIMULATION ####################################################
    # Define the PARAMETERS #########################################
    max_k=50
    num_k=1200
    resolution_side=450 # generated images will be resolution_side x resolution_side
    sim_chunk_ax=550

    # Instatiate simulator and run simulations
    simulator=RingSimulator_Optimizer_GPU( n=1.5, w0=1, a0=1.0, max_k=max_k, num_k=num_k, nx=resolution_side, sim_chunk_x=sim_chunk_ax, sim_chunk_y=sim_chunk_ax)

    os.makedirs(f'./PIPELINE/{experiment_name}/SIMULATIONS/', exist_ok=True)
