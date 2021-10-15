import sys
from GPU_Classes import *
import numpy as np
import json
import cv2

if __name__ == '__main__':
    # PARAMETER SETTINGS ##############################################
    ###################################################################
    ##################################################################
    # 0. GENERAL SETTINGS #############################################
    ################################################################
    experiment_name="USING_INTERPOLATION_IN_iX" # "NOT_USING_INTERPOLATION_IN_iX"
    randomization_seed=666
    image_depth=8 # or 16 bit per pixel

    # 1. SIMULATION ####################################################
    # Define the PARAMETERS #########################################
    # Ring parameters to test (each will be a different simulation)
    reference_theoretical_phiCR=[1,-2]
    reference_R0_s=[25,30]
    reference_w0_s=[5,10]

    problem_theoretical_phiCRs=[2,-3]
    problem_R0_s=reference_R0_s
    problem_w0_s=reference_w0_s

    phiCR_s={'reference':reference_theoretical_phiCR, 'problem':problem_theoretical_phiCRs}
    R0_s={'reference':reference_R0_s, 'problem':problem_R0_s}
    w0_s={'reference':reference_w0_s, 'problem':problem_w0_s}

    resolution_side_nx=100 # generated images will be resolution_side x resolution_side
    # Other parameters
    max_k=50
    num_k=1200
    sim_chunk_ax=210

    # 2. WHITE NOISE ####################################################
    # Define the PARAMETERS #########################################
    number_of_samples_per_sigma={'reference':2, 'problem':2} # they will be simulated once but different noises will be inserted

    noise_sigmas=[10,5,0] # same white noise for reference and problem linked pairs will be used
    #noise_sigmas_reference=[] # the standard deviation of the gaussian noise in each pixel
    #noise_sigmas_problem=[] # in principle the noise in the reference and the problem should be the same

    # 3. SATURATION ####################################################
    saturations_at_relative_intesities=[0.75,1] # capping will be performed at different relative intensities
    # the profile will be multiplied by a factor and then capped (not capped directlty, since that
    # would not replicate the experimental capping)

    # 4. GRAVICENTER iX #################################################
    X=int(resolution_side_nx*1.2/2)
    interpolation_flags={"CUBIC":cv2.INTER_CUBIC, "LANCZOS":cv2.INTER_LANCZOS4}# "LINEAR":cv2.INTER_LINEAR, "AREA":cv2.INTER_AREA, "NEAREST":cv2.INTER_NEAREST}

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

    im_type=np.uint16 if image_depth==16 else np.uint8
    max_intensity=65535 if image_depth==16 else 255
    np.random.seed(randomization_seed)

    # we try to import the paths of the images, in case they have already been partially treated
    try:
        image_paths = json.load(open(f"./PIPELINE/{experiment_name}/{experiment_name}.json"))
    except:
        image_paths={'stage':0, 'simulation':{'reference':[], 'problem':[]}}

    # 1. SIMULATION ####################################################
    simulator=RingSimulator_Optimizer_GPU( n=1.5, a0=1.0, max_k=max_k, num_k=num_k, nx=resolution_side_nx, sim_chunk_x=sim_chunk_ax, sim_chunk_y=sim_chunk_ax)

    for turn in ['reference', 'problem']:
        for phi_CR in phiCR_s[turn]:
            for R0 in R0_s[turn]:
                for w0 in w0_s[turn]:
                    path=f'./PIPELINE/{experiment_name}/SIMULATIONS/nx_{resolution_side_nx}_phiCR_{phi_CR}_R0_{R0}_w0_{w0}' # it should not contain a / in the end, for the rest of the code to work (since the name of the image is extracted from the last part of the directory path)
                    if path not in image_paths['simulation'][turn]:
                        I=simulator.compute_CR_ring( CR_ring_angle=phi_CR, R0_pixels=R0, Z=0, w0_pixels=w0)
                        os.makedirs(path, exist_ok=True)
                        cv2.imwrite(f"{path}/{path.split('/')[-1]}.png", (max_intensity*I).astype(im_type))
                        image_paths['simulation'][turn].append(path)
                        # we save the progess (in order to be able to quit and resume)
                        json.dump(image_paths, open( f"./PIPELINE/{experiment_name}/{experiment_name}.json", "w"))
    if image_paths['stage']==0:
        image_paths['stage']=1
        image_paths['noise']={'reference':[], 'problem':[]}

    # 2. WHITE NOISE ####################################################
    # we take each image and apply the required procedure to each of them
    for turn in ['reference', 'problem']:
        for image_path in image_paths['simulation'][turn]:
            image = cv2.imread(image_path+'/'+image_path.split('/')[-1]+'.png', cv2.IMREAD_ANYDEPTH)
            for sigma in noise_sigmas:
                for empirical_copy in range(number_of_samples_per_sigma[turn]):
                    path=f"{image_path}/WHITE_NOISES/sigma_{sigma}_take_{empirical_copy}_{image_path.split('/')[-1]}"
                    if path not in image_paths['noise'][turn]:
                        if sigma==0:
                            noisy_image = image.astype(np.float64)
                        else:
                            noisy_image = np.abs(image.astype(np.float64)+np.random.normal(loc=0, scale=sigma, size=image.shape))
                        os.makedirs(path, exist_ok=True)
                        cv2.imwrite(f"{path}/{path.split('/')[-1]}.png", (max_intensity*(noisy_image/np.max(noisy_image))).astype(im_type))
                        image_paths['noise'][turn].append(path)
                        # we save the progess (in order to be able to quit and resume)
                        json.dump(image_paths, open( f"./PIPELINE/{experiment_name}/{experiment_name}.json", "w"))
    if image_paths['stage']==1:
        image_paths['stage']=2
        image_paths['saturation']={'reference':[], 'problem':[]}

    # 3. SATURATION ####################################################
    for turn in ['reference', 'problem']:
        for image_path in image_paths['noise'][turn]:
            image = cv2.imread(image_path+'/'+image_path.split('/')[-1]+'.png', cv2.IMREAD_ANYDEPTH)
            for saturation in saturations_at_relative_intesities:
                path=f"{image_path}/SATURATION/satur_{saturation}_{image_path.split('/')[-1]}"
                if path not in image_paths['saturation'][turn]:
                    saturated_image = np.where( image<=(max_intensity*saturation), image, max_intensity)
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(f"{path}/{path.split('/')[-1]}.png", saturated_image)
                    image_paths['saturation'][turn].append(path)
                    # we save the progess (in order to be able to quit and resume)
                    json.dump(image_paths, open( f"./PIPELINE/{experiment_name}/{experiment_name}.json", "w"))
    if image_paths['stage']==2:
        image_paths['stage']=3
        image_paths['iX']={'reference':[], 'problem':[]}


    # 4. GRAVICENTER iX ###############################
    def compute_intensity_gravity_center(image):
        """
            Expects input image to be an array of dimensions [h, w].
            It will return an array of gravity centers [2(h,w)] in pixel coordinates
            Remember that pixel coordinates are set equal to numpy indices

        """
        # image wise total intensity and marginalized inensities for weighted sum
        intensity_in_w = np.sum(image, axis=0) # weights for x [raw_width]
        intensity_in_h = np.sum(image, axis=1) # weights for y [raw_height]
        total_intensity = intensity_in_h.sum()

        # Compute mass center for intensity
        # [2] (h_center,w_center)
        return np.nan_to_num( np.stack(
            (np.dot(intensity_in_h, np.arange(image.shape[0]))/total_intensity,
             np.dot(intensity_in_w, np.arange(image.shape[1]))/total_intensity)
            ) )

    def compute_raw_to_centered_iX(image, X, interpolation_flag):

        g_raw = compute_intensity_gravity_center(image)
        # crop the iamges with size (X+1+X)^2 leaving the gravity center in
        # the central pixel of the image. In case the image is not big enough for the cropping,
        # a 0 padding will be made.
        centered_image = np.zeros( (2*X+1, 2*X+1),  dtype = image.dtype )

        # we round the gravity centers to the nearest pixel indices
        g_index_raw = np.rint(g_raw).astype(int) #[N_images, 2]

        # obtain the slicing indices around the center of gravity
        # TODO -> make all this with a single array operation by stacking the lower and upper in
        # a new axis!!
        # [ 2 (h,w)]
        unclipped_lower = g_index_raw[:]-X
        unclipped_upper = g_index_raw[:]+X+1
        # unclippde could get out of bounds for the indices, so we clip them
        lower_bound = np.clip( unclipped_lower, a_min=0, a_max=image.shape)
        upper_bound = np.clip( unclipped_upper, a_min=0, a_max=image.shape)
        # we use the difference between the clipped and unclipped to get the necessary padding
        # such that the center of gravity is left still in the center of the image
        padding_lower = lower_bound-unclipped_lower
        padding_upper = upper_bound-unclipped_upper

        # crop the image
        centered_image[padding_lower[0]:padding_upper[0] or None,
                                        padding_lower[1]:padding_upper[1] or None ] = \
                      image[lower_bound[0]:upper_bound[0],
                                          lower_bound[1]:upper_bound[1]]
        """
        # We compute the center of gravity of the cropped images, if everything was made allright
        # they should get just centered in the central pixels number X+1 (index X)
        g_centered = compute_intensity_gravity_center(centered_image)

        # We now compute a floating translation of the image so that the gravicenter is exactly
        # centered at pixel (607.5, 607.5) (exact center of the image in pixel coordinates staring
        # form (0,0) and having size (607*2+1)x2), instead of being centered at the beginning of
        # around pixel (607,607) as is now
        translate_vectors = X+0.5-g_centered #[ 2(h,w)]
        T = np.float64([[1,0, translate_vectors[1]], [0,1, translate_vectors[0]]])
        return cv2.warpAffine( centered_image, T, (X*2+1, X*2+1),
                    flags=interpolation_flag) # interpolation method
        """
        return centered_image

    for turn in ['reference', 'problem']:
        for saturation_path in image_paths['saturation'][turn]:
            for interpolation_name, interpolation_flag in interpolation_flags.items():
                image = cv2.imread(saturation_path+'/'+saturation_path.split('/')[-1]+'.png', cv2.IMREAD_ANYDEPTH)
                path=f"{saturation_path}/interpol_{interpolation_name}_iX_{X}_{saturation_path.split('/')[-1]}"
                if path not in image_paths['iX'][turn]:
                    I=compute_raw_to_centered_iX(image.astype(np.float64), X, interpolation_flag)
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(f"{path}/{path.split('/')[-1]}.png",
                        (max_intensity*(I/np.max(I))).astype(im_type))
                    image_paths['iX'][turn].append(path)
                    # we save the progess (in order to be able to quit and resume)
                    json.dump(image_paths, open( f"./PIPELINE/{experiment_name}/{experiment_name}.json", "w"))
    if image_paths['stage']==3:
        image_paths['stage']=4
        image_paths['iX_averaged']={'reference':[], 'problem':[]}

    # 5. COMPUTE AVERAGE IMAGES FROM EACH SATURATED IMAGE SERIES OF THE SAME NOISE ############
    for turn in ['reference', 'problem']:
        for simulation_path in image_paths['simulation'][turn]:
            for saturation in saturations_at_relative_intesities:
                for sigma in noise_sigmas:
                    for interpolation_name in interpolation_flags.keys():
                        average_image=np.zeros( (2*X+1,2*X+1), dtype=np.float64)
                        for empirical_copy in range(number_of_samples_per_sigma[turn]):
                            image_path=f"sigma_{sigma}_take_{empirical_copy}_{simulation_path.split('/')[-1]}"
                            iX_noisy_saturated_take_path=f"{simulation_path}/WHITE_NOISES/{image_path}/SATURATION/satur_{saturation}_{image_path}/interpol_{interpolation_name}_iX_{X}_satur_{saturation}_{image_path}"
                            next_image=cv2.imread(f"{iX_noisy_saturated_take_path}/{iX_noisy_saturated_take_path.split('/')[-1]}.png", cv2.IMREAD_ANYDEPTH)
                            average_image += next_image.astype(np.float64)
                        average_image = average_image/number_of_samples_per_sigma[turn]
                        image_path=f"interpol_{interpolation_name}_iX_{X}_satur_{saturation}_sigma_{sigma}_{simulation_path.split('/')[-1]}"
                        save_path=f"{simulation_path}/WHITE_NOISES/AVERAGES/{image_path}"
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite( f"{save_path}/{save_path.split('/')[-1]}.png",
                            (max_intensity*(average_image/np.max(average_image))).astype(im_type))
                        image_paths['iX_averaged'][turn].append(save_path)
                        json.dump(image_paths, open( f"./PIPELINE/{experiment_name}/{experiment_name}.json", "w"))

    if image_paths['stage']==4:
        image_paths['stage']=5


    # 6. POLARIZATION RELATIVE ANGLES ###################################
    # Mirror with affine interpolation & Rotation Algorithms will be employed
    # Each using both Fibonacci and Quadratic Fit Search
    # Results will be gathered in a table and outputed as an excel csv
    # Mock Image Loader
    # Computar el angulo de cada uno en un dataframe donde una de las entradas sea results y haya un result per fibo qfs y per rotation y mirror affine. Y luego procesar en un 7º paso estos angulos para obtener los angulos relativos etc y perhaps hacer tablucha con ground truth menos el resulting delta angle medido por el algoritmo
    image_loader = Image_Manager(mode=X, interpolation_flag=None)
    for simulation_path in image_paths['simulation'][turn]:
        for saturation in saturations_at_relative_intesities:
            for sigma in noise_sigmas:
                for interpolation_name in interpolation_flags.keys():
                    average_image=np.zeros( (2*X+1,2*X+1), dtype=np.float64)
                    for empirical_copy in range(number_of_samples_per_sigma[turn]):
                        image_path=f"sigma_{sigma}_take_{empirical_copy}_{simulation_path.split('/')[-1]}"
                        iX_noisy_saturated_take_path=f"{simulation_path}/WHITE_NOISES/{image_path}/SATURATION/satur_{saturation}_{image_path}/interpol_{interpolation_name}_iX_{X}_satur_{saturation}_{image_path}"
                        next_image=cv2.imread(f"{iX_noisy_saturated_take_path}/{iX_noisy_saturated_take_path.split('/')[-1]}.png", cv2.IMREAD_ANYDEPTH)

    # Define the ROTATION ALGORITHM
    rotation_algorithm = Rotation_Algorithm(self.image_loader,
        eval(self.theta_min_R.text()), eval(self.theta_max_R.text()),
        self.choose_interpolation_falg(self.interpolation_alg_opt),
        float(self.initial_guess_delta_rad.text()), self.use_exact_grav_R.isChecked())

    # Define the Affine Mirror algorithm
    mirror_algorithm = Mirror_Flip_Algorithm(self.image_loader,
        eval(self.theta_min_M.text()), eval(self.theta_max_M.text()),
        self.choose_interpolation_falg(self.interpolation_alg_opt),
        float(self.initial_guess_delta_rad.text()), method, self.left_vs_right_M.isChecked(), self.use_exact_grav_M.isChecked())

    # Get arguments and run algorithm depending on the chosen stuff
    rotation_algorithm.fibonacci_ratio_search(
            float(self.precision_fib_rad.text()), int(self.max_points_fib.text()),
            float(self.cost_tolerance_fib.text())
        )
    to_benchmark_dict(benchu, rotation_algorithm, image_names, "R - Fibonacci Ratio", ground_truths)

    rotation_algorithm.reInitialize(self.image_loader)
    rotation_algorithm.quadratic_fit_search(
            float(self.precision_quad_rad.text()),
            int(self.max_it_quad.text()),
            float(self.cost_tolerance_quad.text())
        )
    to_benchmark_dict(benchu, rotation_algorithm, image_names, "R - Quadratic Fit", ground_truths)


    # 6. OUTPUT RESULTS INTO A LATEX? INTO AN EXCEL WITH IMAGES (GIFS) WOULD BE FANTASTIC
    # Instatiate simulator and run simulations
