import os
import sys
from GPU_Classes import *
from Image_Manager import *
from Polarization_Obtention_Algorithms import Rotation_Algorithm, Mirror_Flip_Algorithm
import numpy as np
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # PARAMETER SETTINGS ##############################################
    ###################################################################
    ##################################################################
    # 0. GENERAL SETTINGS #############################################
    ################################################################
    experiment_name="fix_R0_vary_w0_" # "NOT_USING_INTERPOLATION_IN_iX" # "RECENTERING_AVERAGE_IMAGE_TO_iX_USING_INTERPOLATION" # "RECENTERING_AVERAGE_IMAGE_TO_iX_NOT_USING_INTERPOLATION" # los 4 con 540 y los 4 con el doble de resolucion y ver que sacamos de los resultados - note that this means only two simulation rounds are necessary
    randomization_seed=666
    image_depth=8 # or 16 bit per pixel
    use_interpolation=False
    recenter_average_image=False

    # 1. SIMULATION ####################################################
    # Define the PARAMETERS #########################################
    # Ring parameters to test (each will be a different simulation)
    reference_theoretical_phiCR=np.linspace(0,2*np.pi-0.1,40)
    reference_R0_s=[167] # in pxels
    rho_0s=np.array([2,3,4,5,6,7,9,11,14,20])
    reference_w0_s=167/rho_0s


    problem_theoretical_phiCRs=np.linspace(0.1,2*np.pi+0.05,41)
    problem_R0_s=reference_R0_s
    problem_w0_s=reference_w0_s

    phiCR_s={'reference':reference_theoretical_phiCR, 'problem':problem_theoretical_phiCRs}
    R0_s={'reference':reference_R0_s, 'problem':problem_R0_s}
    w0_s={'reference':reference_w0_s, 'problem':problem_w0_s}

    resolution_side_nx=540 # generated images will be resolution_side x resolution_side
    # Other parameters
    max_k=50
    num_k=1200
    sim_chunk_ax=540

    # 4. GRAVICENTER iX and PROFILES ######################################
    X=int(resolution_side_nx*1.4/2)
    interpolation_flags={"CUBIC":cv2.INTER_CUBIC, "LANCZOS":cv2.INTER_LANCZOS4}# "LINEAR":cv2.INTER_LINEAR, "AREA":cv2.INTER_AREA, "NEAREST":cv2.INTER_NEAREST}
    if use_interpolation is False:
        interpolation_flags={"NONE":None}

    # 5. POLARIZATION RELATIVE ANGLES ###################################
    # Mirror with affine interpolation & Rotation Algorithms will be employed
    # Each using both Fibonacci and Quadratic Fit Search
    # Results will be gathered in a table and outputed as an excel csv
    theta_min_Rot=-np.pi
    theta_max_Rot=np.pi
    theta_min_Mir=0
    theta_max_Mir=np.pi
    initial_guess_delta_rad=0.1
    use_exact_gravicenter=True
    precision_quadratic=1e-9
    max_it_quadratic=100
    cost_tolerance_quadratic=1e-12
    precision_fibonacci=1e-9
    max_points_fibonacci=100
    cost_tolerance_fibonacci=1e-12

    # 6. OUTPUT RESULTS INTO A LATEX? INTO AN EXCEL WITH IMAGES (GIFS) WOULD BE FANTASTIC
    deg_or_rad="rad" # for the final outputs

    experiment_name=f"{experiment_name}_nx_{resolution_side_nx}_iX_{X}_angles_{deg_or_rad}"
    ##################################################################
    ##################################################################

    # PIPELINE EXECUTION!!! ##########################################
    ##################################################################
    # 0. GENERAL SETTINGS ############################################
    ##################################################################
    os.makedirs(f'./OUTPUT/PIPELINE/{experiment_name}/SIMULATIONS/', exist_ok=True)

    im_type=np.uint16 if image_depth==16 else np.uint8
    max_intensity=65535 if image_depth==16 else 255
    np.random.seed(randomization_seed)

    # we try to import the paths of the images, in case they have already been partially treated
    try:
        image_paths = json.load(open(f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json"))
    except:
        image_paths={'stage':0, 'simulation':{'reference':[], 'problem':[]}}

    # 1. SIMULATION ####################################################
    simulator=RingSimulator_Optimizer_GPU( n=1.5, a0=1.0, max_k=max_k, num_k=num_k, nx=resolution_side_nx, sim_chunk_x=sim_chunk_ax, sim_chunk_y=sim_chunk_ax)
    i=1
    for turn in ['reference', 'problem']:
        for phi_CR in phiCR_s[turn]:
            for R0 in R0_s[turn]:
                for w0 in w0_s[turn]:
                    path=f'./OUTPUT/PIPELINE/{experiment_name}/SIMULATIONS/nx_{resolution_side_nx}_phiCR_{phi_CR}_R0_{R0}_w0_{w0}' # it should not contain a / in the end, for the rest of the code to work (since the name of the image is extracted from the last part of the directory path)
                    if path not in image_paths['simulation'][turn]:
                        I=simulator.compute_CR_ring( CR_ring_angle=phi_CR, R0_pixels=R0, Z=0, w0_pixels=w0)
                        os.makedirs(path, exist_ok=True)
                        cv2.imwrite(f"{path}/{path.split('/')[-1]}.png", (max_intensity*I/I.max()).astype(im_type))
                        image_paths['simulation'][turn].append(path)
                        # we save the progess (in order to be able to quit and resume)
                        json.dump(image_paths, open( f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json", "w"))
                        print(f"{i}-th Simulated")
                        i+=1
    if image_paths['stage']==0:
        image_paths['stage']=1
        image_paths['noise']={'reference':[], 'problem':[]}
    print("1. Simulations Finished!\n")
    '''
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
                        json.dump(image_paths, open( f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json", "w"))
    if image_paths['stage']==1:
        image_paths['stage']=2
        image_paths['saturation']={'reference':[], 'problem':[]}
    print("2. Noisy Images generated!\n")

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
                    json.dump(image_paths, open( f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json", "w"))
    '''
    if image_paths['stage']<=2:
        image_paths['stage']=3
        image_paths['iX']={'reference':[], 'problem':[]}
    print("3. Saturated Images Computed!\n")

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

    def compute_raw_to_centered_iX(image, X, interpolation_flag=None):

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
        if interpolation_flag==None:
            return centered_image
        else:
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


    for turn in ['reference', 'problem']:
        for saturation_path in image_paths['simulation'][turn]:
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
                    json.dump(image_paths, open( f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json", "w"))
    if image_paths['stage']==3:
        image_paths['stage']=4
        image_paths['iX_averaged']={'reference':[], 'problem':[]}
    print("4. iX images computed!\n")
    '''
    # 5. COMPUTE AVERAGE IMAGES FROM EACH SATURATED IMAGE SERIES OF THE SAME NOISE and #######
    #    compute PROFILES FOR THEM ###########################################################
    def initialize_plot_blit(sample_image):
        fig = plt.figure(figsize=(2*6, 2*6))
        axes=fig.subplots(2,2)

        cm=axes[0, 0].imshow(sample_image, cmap='viridis', animated=True)
        axes[0,0].grid(True)

        prof_x=np.sum(sample_image, axis=0)
        prof_y=np.sum(sample_image, axis=1)
        scat1, = axes[0,1].plot([], 'o', markersize=2, label=f'Intensity profile in y', animated=True)
        axes[0,1].set_ylim((0,len(prof_y)))
        axes[0,1].set_xlim(0,1.2*prof_y.max())
        axes[0,1].invert_yaxis()
        scat2, = axes[1,0].plot([],'o', markersize=2, label=f'Intensity profile in y', animated=True)
        axes[1,0].set_xlim((0,len(prof_x)))
        axes[1,0].set_ylim(0,1.2*prof_x.max())

        axes[1,0].invert_yaxis()
        axes[0,0].set_xlabel("x (pixels)")
        #axes[0,0].set_ylabel("y (pixels)")
        axes[0,1].set_xlabel("Cummulative Intensity")
        axes[0,1].set_ylabel("y (pixels)")
        axes[1,0].set_ylabel("Cummulative Intensity")
        axes[1,0].set_xlabel("x (pixels)")
        axes[1,0].grid(True)
        axes[0,1].grid(True)
        axes[1,1].set_visible(False)
        ax = fig.add_subplot(224, projection='3d')
        X,Y = np.meshgrid(np.arange(len(prof_y)),np.arange(len(prof_x)))
        #fig.suptitle(f"Intesity Profiles for Image\n{output_full_path.split('/')[-1]}")
        cbax=fig.add_axes([0.54,0.05,0.4,0.01])
        fig.colorbar(cm, ax=axes[0,0], cax=cbax, orientation='horizontal')
        #for theta in np.linspace(0, 360, 40):
        #ax.clear()
        plot3d=ax.plot_surface(X, Y, np.array([[0]]), rcount=int(len(prof_y)*plot_3d_finnes), ccount=int(len(prof_x)*plot_3d_finnes), cmap='viridis', animated=True) # rstride=1, cstride=1, linewidth=0
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Intensity')
        ax.set_zlim(-0.078*np.max(sample_image), np.max(sample_image))
        ax.set_title("Image intensity 3D plot")
        theta=25
        phi=30
        ax.view_init(phi, theta)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.3, 1.3, 1.3, 1]))

        # cache the background
        axbackground1 = fig.canvas.copy_from_bbox(axes[0,0].bbox)
        axbackground2 = fig.canvas.copy_from_bbox(axes[0,1].bbox)
        axbackground3 = fig.canvas.copy_from_bbox(axes[1,0].bbox)
        axbackground4 = fig.canvas.copy_from_bbox(ax.bbox)
        axbackgrounds=[axbackground1, axbackground2, axbackground3, axbackground4]
        #fig.canvas.draw()
        plt.savefig(f"./OUTPUT/PIPELINE/{experiment_name}/Background.png")
        print(f"BACKGROUND")
        return fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds, X,Y

    def compute_profiles(image, output_full_path, fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,X,Y):
        prof_x=np.sum(image, axis=0)
        prof_y=np.sum(image, axis=1)
        # set the new data
        cm.set_data(image)
        scat1.set_data(prof_y,np.arange(len(prof_y)))
        scat2.set_data(np.arange(len(prof_x)), prof_x)
        #plot3d.set_zdata(image)
        ax.clear()
        ax.plot_surface(X, Y, image.T, rcount=int(len(prof_y)*plot_3d_finnes), ccount=int(len(prof_x)*plot_3d_finnes), cmap='viridis')
        # restore the background
        for i in range(4):
            fig.canvas.restore_region(axbackgrounds[i])
        # redraw the points
        axes[0,0].draw_artist(cm)
        axes[0,1].draw_artist(scat1)
        axes[1,0].draw_artist(scat2)
        ax.draw_artist(plot3d)
        # fill in the axes rectangle
        fig.canvas.blit(axes[0,0].bbox)
        fig.canvas.blit(axes[0,1].bbox)
        fig.canvas.blit(axes[1,0].bbox)
        fig.canvas.blit(ax.bbox)

        fig.suptitle(f"Intesity Profiles for Image\n{output_full_path.split('/')[-1]}")
        fig.savefig(output_full_path)
        #fig.canvas.flush_events()
        #files_for_gif.append(f"{dirpath}/temp/PROFILES_theta_{theta}_{filename}")
        print(f"ploted {output_full_path}")

    profile_plot_initialized=False
    for turn in ['reference', 'problem']:
        for simulation_path in image_paths['simulation'][turn]:
            for saturation in saturations_at_relative_intesities:
                for sigma in noise_sigmas:
                    for interpolation_name, interpolation_flag in interpolation_flags.items():
                        average_image=np.zeros( (2*X+1,2*X+1), dtype=np.float64)
                        for empirical_copy in range(number_of_samples_per_sigma[turn]):
                            image_path=f"sigma_{sigma}_take_{empirical_copy}_{simulation_path.split('/')[-1]}"
                            iX_noisy_saturated_take_path=f"{simulation_path}/WHITE_NOISES/{image_path}/SATURATION/satur_{saturation}_{image_path}/interpol_{interpolation_name}_iX_{X}_satur_{saturation}_{image_path}"
                            next_image=cv2.imread(f"{iX_noisy_saturated_take_path}/{iX_noisy_saturated_take_path.split('/')[-1]}.png", cv2.IMREAD_ANYDEPTH)
                            average_image += next_image.astype(np.float64)
                            # compute the profile
                            if empirical_copy==0:
                                if profile_plot_initialized==False:
                                    fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,Xg,Y = initialize_plot_blit(next_image)
                                    profile_plot_initialized=True
                                compute_profiles(next_image, f"{iX_noisy_saturated_take_path}/PROFILES_{iX_noisy_saturated_take_path.split('/')[-1]}.png", fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,Xg,Y)

                        average_image = average_image/number_of_samples_per_sigma[turn]
                        # in theory, the average image should readily be centered in the gravicenter but we can force it to be so:
                        if recenter_average_image:
                            average_image = compute_raw_to_centered_iX(average_image, X, interpolation_flag)
                        image_path=f"recenter_{recenter_average_image}_interpol_{interpolation_name}_iX_{X}_satur_{saturation}_sigma_{sigma}_{simulation_path.split('/')[-1]}"
                        save_path=f"{simulation_path}/WHITE_NOISES/AVERAGES/{image_path}"
                        os.makedirs(save_path, exist_ok=True)
                        average_image=(max_intensity*(average_image/np.max(average_image))).astype(im_type)
                        cv2.imwrite( f"{save_path}/{save_path.split('/')[-1]}.png", average_image)
                        # compute the profiles
                        compute_profiles(average_image, f"{save_path}/PROFILES_{save_path.split('/')[-1]}.png", fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,Xg,Y)
                        image_paths['iX_averaged'][turn].append(save_path)
                        json.dump(image_paths, open( f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json", "w"))
    '''
    if image_paths['stage']<=4:
        image_paths['stage']=5
        json.dump(image_paths, open( f"./OUTPUT/PIPELINE/{experiment_name}/STRUCTURE_{experiment_name}.json", "w"))

    print("5. Averge Images Computed!\n")

    # 6. POLARIZATION RELATIVE ANGLES ###################################
    # Mirror with affine interpolation & Rotation Algorithms will be employed
    # Each using both Fibonacci and Quadratic Fit Search
    # Results will be gathered in a table and outputed as an excel csv
    # Mock Image Loader
    # Computar el angulo de cada uno en un dataframe donde una de las entradas sea results y haya un result per fibo qfs y per rotation y mirror affine. Y luego procesar en un 7º paso estos angulos para obtener los angulos relativos etc y perhaps hacer tablucha con ground truth menos el resulting delta angle medido por el algoritmo
    image_loader = Image_Manager(mode=X, interpolation_flag=None)
    # Define the ROTATION ALGORITHM
    rotation_algorithm = Rotation_Algorithm(image_loader,
        theta_min_Rot, theta_max_Rot, None,
        initial_guess_delta_rad, use_exact_gravicenter, initialize_it=False)

    # Define the Affine Mirror algorithm
    mirror_algorithm = Mirror_Flip_Algorithm(image_loader,
        theta_min_Mir, theta_max_Mir, None,
        initial_guess_delta_rad, method="aff", left_vs_right=True, use_exact_gravicenter=use_exact_gravicenter, initialize_it=False)
    # A dictionary to gather all the resulting angles for each image
    try:
        individual_image_results = json.load(open(f"./OUTPUT/PIPELINE/{experiment_name}/RAW_RESULTS_{experiment_name}.json"))
    except:
        individual_image_results = {'is_reference':[], 'Image_Name':[], 'theoretical_phiCR':[], 'R0':[], 'w0':[], 'rho0':[], 'interpolation':[], 'polarization_method':[], 'optimization_1d':[], 'found_phiCR':[], 'predicted_opt_precision':[] }

    def to_result_dict(result_dict, im_names, alg, alg_name, opt_name, is_reference):
        for key, name in zip(alg.times.keys(), im_names):
            result_dict['is_reference'].append(is_reference)
            result_dict['Image_Name'].append(name)
            result_dict['theoretical_phiCR'].append(float(name.split("phiCR_")[1].split("_")[0]))
            result_dict['R0'].append(name.split("R0_")[1].split("_")[0])
            result_dict['w0'].append(name.split("w0_")[1].split("_")[0])
            result_dict['rho0'].append(float(result_dict['R0'][-1])/float(result_dict['w0'][-1]))
            result_dict['interpolation'].append(name.split("interpol_")[1].split("_")[0])
            result_dict['polarization_method'].append(alg_name)
            result_dict['optimization_1d'].append(opt_name)
            result_dict['found_phiCR'].append(alg.angles[key])
            result_dict['predicted_opt_precision'].append(alg.precisions[key])


    for turn in ['reference', 'problem']:
        for simulation_path in image_paths['simulation'][turn]:
            for interpolation_name, interpolation_flag in interpolation_flags.items():
                image_container=np.zeros( (1, 2*X+1, 2*X+1), dtype=np.float64)
                image_names=[]
                image_path=f"{simulation_path.split('/')[-1]}"
                iX_simulated=f"{simulation_path}/interpol_{interpolation_name}_iX_{X}_{image_path}"
                next_image=cv2.imread(f"{iX_simulated}/{iX_simulated.split('/')[-1]}.png", cv2.IMREAD_ANYDEPTH)
                image_container[0]=next_image.astype(np.float64)
                image_names.append(iX_simulated.split('/')[-1])

                if not set(image_names).issubset(individual_image_results["Image_Name"]): # then must compute
                    # charge the image loader:
                    image_loader.import_converted_images_as_array(image_container, image_names)
                    # Execute the Rotation and Mirror Algorithms:
                    # ROTATION ######
                    # the interpolation algorithm used in case we disbale its usage for the iX image obtention will be the Lanczos one
                    rotation_algorithm.interpolation_flag=interpolation_flag if interpolation_flag is not None else cv2.INTER_LANCZOS4
                    rotation_algorithm.reInitialize(image_loader)
                    rotation_algorithm.quadratic_fit_search(precision_quadratic, max_it_quadratic, cost_tolerance_quadratic)
                    to_result_dict(individual_image_results, image_names, rotation_algorithm, "Rotation", "Quadratic", True if turn=="reference" else False)
                    rotation_algorithm.reInitialize(image_loader)
                    rotation_algorithm.fibonacci_ratio_search(precision_fibonacci, max_points_fibonacci, cost_tolerance_fibonacci)
                    to_result_dict(individual_image_results, image_names, rotation_algorithm, "Rotation", "Fibonacci", True if turn=="reference" else False)

                    # MIRROR #######
                    mirror_algorithm.interpolation_flag=interpolation_flag if interpolation_flag is not None else cv2.INTER_LANCZOS4
                    mirror_algorithm.reInitialize(image_loader)
                    mirror_algorithm.quadratic_fit_search(precision_quadratic, max_it_quadratic, cost_tolerance_quadratic)
                    to_result_dict(individual_image_results, image_names, rotation_algorithm, "Mirror", "Quadratic", True if turn=="reference" else False)
                    mirror_algorithm.reInitialize(image_loader)
                    mirror_algorithm.fibonacci_ratio_search(precision_fibonacci, max_points_fibonacci, cost_tolerance_fibonacci)
                    to_result_dict(individual_image_results, image_names, rotation_algorithm, "Mirror", "Fibonacci", True if turn=="reference" else False)

                    json.dump(individual_image_results, open( f"./OUTPUT/PIPELINE/{experiment_name}/RAW_RESULTS_{experiment_name}.json", "w"))

    print("6. Raw Results Computed!\n")

    # 7. PROCESS FINAL RESULTS ##########################################
    def angle_to_pi_pi(angle): # convert any angle to range ()-pi,pi]
        angle= angle%(2*np.pi) # take it to [-2pi, 2pi]
        return angle-np.sign(angle)*2*np.pi if abs(angle)>np.pi else angle

    def num_of_zeros(n): # To count the number of zero decimals before non-zeros
        s = '{:.16f}'.format(n).split('.')[1]
        return len(s) - len(s.lstrip('0'))
    cwd = os.getcwd() # working directory
    def make_hyperlink(image_name): # for inserting direct link to image paths
        url = f"./SIMULATIONS/nx_{image_name.split('nx_')[-1]}/interpol_{image_name.split('interpol_')[-1]}/interpol_{image_name.split('interpol_')[-1]}.png"
        return f"=HYPERLINK(\"{url}\", \"link\")"

    conv=1 if deg_or_rad=="rad" else 180/np.pi # conversion factor

    try:
        final_results = json.load(open(f"./OUTPUT/PIPELINE/{experiment_name}/FINAL_RESULTS_{experiment_name}.json"))
    except:
        final_results = { 'R0':[],'w0':[], 'rho0':[],'th_phiCR_ref':[],'th_phiCR_prob':[], 'interpolation':[], 'min_abs_theoretical_error':[], 'best_correct_decimals':[], 'best_algorithm':[], 'ref_im_link':[], 'prob_im_link':[], 'mirror_fibo':[], 'mirror_quad':[], 'rotation_fibo':[], 'rotation_quad':[] }

    # def to_final_dict_results(result_dict, raw_result_dict):
    # we will use a pandas dataframe for it is easier to manipulate
    raw_results = pd.DataFrame.from_dict(individual_image_results)

    # what I will do here is veery inefficient, but well, f* that xD
    pd.options.display.max_colwidth = 100
    for ref_group_tuple, reference_df in raw_results[raw_results["is_reference"]==True].groupby(["R0", "w0", "theoretical_phiCR", 'interpolation']):
        # we will avoid experiments crossing sigma, saturation, interpolation and average before or after
        for prob_group_tuple, problem_df in raw_results[(raw_results["is_reference"]==False) & (raw_results["R0"]==ref_group_tuple[0]) & (raw_results["w0"]==ref_group_tuple[1]) & (raw_results["interpolation"]==ref_group_tuple[3])].groupby(["R0", "w0", "theoretical_phiCR", 'interpolation']):

            final_results["R0"].append(ref_group_tuple[0])
            final_results["w0"].append(ref_group_tuple[1])
            final_results["rho0"].append(problem_df["rho0"].iloc[0])
            final_results["th_phiCR_ref"].append(ref_group_tuple[2])
            final_results["th_phiCR_prob"].append(prob_group_tuple[2])
            final_results["interpolation"].append(ref_group_tuple[3])

            final_results["mirror_fibo"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Mirror") & (problem_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Mirror") & (reference_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0])*conv)
            final_results["mirror_quad"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Mirror") & (problem_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Mirror") & (reference_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0])*conv)
            final_results["rotation_fibo"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Rotation") & (problem_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Rotation") & (reference_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0])*conv)
            final_results["rotation_quad"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Rotation") & (problem_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Rotation") & (reference_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0])*conv)


            ground_truth_relative_angle=angle_to_pi_pi(final_results["th_phiCR_prob"][-1]-final_results["th_phiCR_ref"][-1])
            theoretical_errors=np.abs(np.array([ final_results["mirror_fibo"][-1], final_results["mirror_quad"][-1],  final_results["rotation_fibo"][-1], final_results["rotation_quad"][-1]])-ground_truth_relative_angle )
            final_results["min_abs_theoretical_error"].append( theoretical_errors.min()  )
            final_results['best_correct_decimals'].append( num_of_zeros(theoretical_errors.min()) )
            final_results["best_algorithm"].append( ["mirror_fib", "mirror_quad", "rotation_fib", "rotation_quad"][theoretical_errors.argmin()] )
            final_results["ref_im_link"].append(make_hyperlink(reference_df["Image_Name"].iloc[0]))
            final_results["prob_im_link"].append(make_hyperlink(problem_df["Image_Name"].iloc[0]))



    json.dump(final_results, open( f"./OUTPUT/PIPELINE/{experiment_name}/FINAL_RESULTS_{experiment_name}.json", "w"))
    print("7. Final Results Computed!\n")

    # 8. OUTPUT RESULTS INTO AN EXCEL WITH IMAGES (GIFS) ################
    from styleframe import StyleFrame

    final_results_df = pd.DataFrame.from_dict(final_results)

    writer = StyleFrame.ExcelWriter(f"./OUTPUT/PIPELINE/{experiment_name}/EXCEL_FINAL_RESULTS_{experiment_name}.xlsx")

    # Convert the dataframe to an XlsxWriter Excel object.
    StyleFrame.A_FACTOR=10
    StyleFrame.P_FACTOR=0.9
    StyleFrame(final_results_df.sort_values(by=['min_abs_theoretical_error'])).set_row_height(1,50).to_excel(writer, best_fit=list(final_results_df.columns)[:9]+list(final_results_df.columns)[11:], sheet_name='Experiment Results by Min Error', index=False,  float_format="%.12f")
    StyleFrame(final_results_df.sort_values(by=['R0','w0','th_phiCR_ref','th_phiCR_prob','min_abs_theoretical_error','interpolation'], ascending=False)).set_row_height(1,50).to_excel(writer, best_fit=list(final_results_df.columns)[:9]+list(final_results_df.columns)[11:], sheet_name='Experiment Results by Properties', index=False,  float_format="%.12f")
    StyleFrame(raw_results).set_row_height(1,50).to_excel(writer, best_fit=list(raw_results.columns), sheet_name='Raw Results per Image', index=False,  float_format="%.12f")
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    #import xlsxwriter as xl
    #excel_file = xlsxwriter.Workbook(f"./OUTPUT/PIPELINE/{experiment_name}/EXCEL_FINAL_RESULTS_{experiment_name}.xlsx")

    #raw_results = pd.DataFrame.from_dict(individual_image_results)
    #worksheet1 = excel_file.add_worksheet("Experiment Results")


    #worksheet2 = excel_file.add_worksheet("Raw Results per Image")
    #for row in range(raw_results.shape[0]):
