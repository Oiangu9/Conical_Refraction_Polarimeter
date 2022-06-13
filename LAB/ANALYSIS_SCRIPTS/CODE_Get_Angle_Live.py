import os

import sys
from SOURCE.CLASS_CODE_GPU_Classes import *
from SOURCE.CLASS_CODE_Image_Manager import *
from SOURCE.CLASS_CODE_Polarization_Obtention_Algorithms import Rotation_Algorithm, Mirror_Flip_Algorithm
import numpy as np
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glob


def run_angle_live(output_path, ground_truth, saturation, silhouette=False):

    reference_directory=output_path+f"/Reference"
    problem_directory=output_path+f"/Problem"
    image_shortest_side=540

    #ground_truth=-13.85 # if deg selected then input ground truth in deg, else in rad
    pol_or_CR="pol"
    deg_or_rad="deg" # for the final outputs
    image_depth=8 # or 16 bit per pixel
    randomization_seed=666
    use_interpolation=False
    recenter_average_image=False

    # 4. GRAVICENTER iX and PROFILES ######################################
    X=int(image_shortest_side*1.4/2)
    interpolation_flags={"CUBIC":cv2.INTER_CUBIC, "LANCZOS":cv2.INTER_LANCZOS4}# "LINEAR":cv2.INTER_LINEAR, "AREA":cv2.INTER_AREA, "NEAREST":cv2.INTER_NEAREST}
    plot_3d_finnes=0.3 # value that should go in (0,1]. 1 means all the pixels will be ploted in the 3d plot, 0.5 only half of them



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
    precision_quadratic=1e-10
    max_it_quadratic=100
    cost_tolerance_quadratic=1e-14
    precision_fibonacci=1e-10
    max_points_fibonacci=100
    cost_tolerance_fibonacci=1e-14




    experiment_name=f"{reference_directory.split('/')[-1]}_AND_{reference_directory.split('/')[-1]}_WITH_iX_{X}_angles_{deg_or_rad}_rel_satur_{saturation}"
    ##################################################################
    ##################################################################
    im_type=np.uint16 if image_depth==16 else np.uint8
    max_intensity=65535 if image_depth==16 else 255
    np.random.seed(randomization_seed)
    if use_interpolation is False:
        interpolation_flags={"NONE":None}
    polCR=1 if pol_or_CR=='CR' else 0.5

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

    ref_images=glob.glob(f"{reference_directory}/*.png")
    pb_images=glob.glob(f"{problem_directory}/*.png")

    for mother_dir, image_list in zip([reference_directory, problem_directory],[ref_images, pb_images]):
        for image_path in image_list:
            for interpolation_name, interpolation_flag in interpolation_flags.items():
                image = (cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)).astype(np.float64)
                # normalize the image
                image = max_intensity*image/image.max()
                # apply saturation or silhouette
                if silhouette==False:
                    image = np.where( image<=(max_intensity*saturation), image, max_intensity*saturation)
                else:
                    image = np.where( image<=(max_intensity*saturation), 0, max_intensity)
                I=compute_raw_to_centered_iX(image, X, interpolation_flag)
                os.makedirs(f"{mother_dir}/iX", exist_ok=True)
                cv2.imwrite(f"{mother_dir}/iX/iX_{X}_{image_path.split('/')[-1]}.png",
                        (max_intensity*(I/np.max(I))).astype(im_type))
    iX_ref_images=set(glob.glob(f"{reference_directory}/iX/*.png"))-set(glob.glob(f"{reference_directory}/iX/AVERAGE*"))-set(glob.glob(f"{reference_directory}/iX/PROFILE*"))
    iX_pb_images=set(glob.glob(f"{problem_directory}/iX/*.png"))-set(glob.glob(f"{problem_directory}/iX/AVERAGE*"))-set(glob.glob(f"{problem_directory}/iX/PROFILE*"))

    print("1. Gravicenters computed!\n")


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
        theta=-40
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
        plt.savefig(f"{output_path}/Background.png")
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

    profile_plot_initialized=False
    for mother_dir, image_list in zip([reference_directory, problem_directory],[iX_ref_images, iX_pb_images]):
        average_image=np.zeros( (2*X+1,2*X+1), dtype=np.float64)
        for image_path in image_list:
            next_image=cv2.imread(f"{image_path}", cv2.IMREAD_ANYDEPTH)
            average_image += next_image.astype(np.float64)
            # compute the profile of the each image
            if profile_plot_initialized==False:
                fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,Xg,Y = initialize_plot_blit(next_image)
                profile_plot_initialized=True
            compute_profiles(next_image, f"{mother_dir}/iX/PROFILE_{image_path.split('/')[-1]}", fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,Xg,Y)

        average_image = average_image/len(image_list)
        # in theory, the average image should readily be centered in the gravicenter but we can force it to be so:
        if recenter_average_image:
            average_image = compute_raw_to_centered_iX(average_image, X, None)
        average_image=(max_intensity*(average_image/np.max(average_image))).astype(im_type)
        cv2.imwrite( f"{mother_dir}/iX/AVERAGE_{mother_dir.split('/')[-1]}.png", average_image)
        # compute the profiles
        compute_profiles(average_image, f"{mother_dir}/iX/PROFILE_AVERAGE_{mother_dir.split('/')[-1]}.png", fig, cm, scat1, scat2, plot3d, axes, ax, axbackgrounds,Xg,Y)


    print("2. Averge Images and Profiles Computed!\n")

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

    individual_image_results = {'is_reference':[], 'Image_Name':[], 'averaged_before_or_after':[],  'polarization_method':[], 'optimization_1d':[], 'found_phiCR':[], 'predicted_opt_precision':[] }

    def to_result_dict(result_dict, im_names, alg, alg_name, opt_name, is_reference):
        for key, name in zip(alg.times.keys(), im_names):
            result_dict['is_reference'].append(is_reference)
            result_dict['Image_Name'].append(name)
            if 'AVERAGE' in name:
                result_dict['averaged_before_or_after'].append('B')
            else:
                result_dict['averaged_before_or_after'].append('A')
            result_dict['polarization_method'].append(alg_name)
            result_dict['optimization_1d'].append(opt_name)
            result_dict['found_phiCR'].append(alg.angles[key])
            result_dict['predicted_opt_precision'].append(alg.precisions[key])
    k=0
    for mother_dir, image_list in zip([reference_directory, problem_directory],[iX_ref_images, iX_pb_images]):
        image_container=np.zeros( (len(image_list)+1, 2*X+1, 2*X+1), dtype=np.float64)
        image_names=[]
        # charge the different noise takes
        for i,image_path in enumerate(image_list):
            next_image=cv2.imread(f"{image_path}", cv2.IMREAD_ANYDEPTH)
            image_container[i]=next_image.astype(np.float64)
            image_names.append(image_path.split('/')[-1])
        # charge the average image
        next_image=cv2.imread( f"{mother_dir}/iX/AVERAGE_{mother_dir.split('/')[-1]}.png", cv2.IMREAD_ANYDEPTH)
        image_container[len(image_list)]=next_image.astype(np.float64)
        image_names.append(f"AVERAGE_{mother_dir.split('/')[-1]}")

        # charge the image loader:
        image_loader.import_converted_images_as_array(image_container, image_names)
        # Execute the Rotation and Mirror Algorithms:
        # ROTATION ######
        # the interpolation algorithm used in case we disbale its usage for the iX image obtention will be the Lanczos one
        rotation_algorithm.interpolation_flag=interpolation_flag if interpolation_flag is not None else cv2.INTER_CUBIC
        rotation_algorithm.reInitialize(image_loader)
        rotation_algorithm.quadratic_fit_search(precision_quadratic, max_it_quadratic, cost_tolerance_quadratic)
        to_result_dict(individual_image_results, image_names, rotation_algorithm, "Rotation", "Quadratic", True if k==0 else False)
        rotation_algorithm.reInitialize(image_loader)
        rotation_algorithm.fibonacci_ratio_search(precision_fibonacci, max_points_fibonacci, cost_tolerance_fibonacci)
        to_result_dict(individual_image_results, image_names, rotation_algorithm, "Rotation", "Fibonacci", True if k==0 else False)

        # MIRROR #######
        mirror_algorithm.interpolation_flag=interpolation_flag if interpolation_flag is not None else cv2.INTER_CUBIC
        mirror_algorithm.reInitialize(image_loader)
        mirror_algorithm.quadratic_fit_search(precision_quadratic, max_it_quadratic, cost_tolerance_quadratic)
        to_result_dict(individual_image_results, image_names, rotation_algorithm, "Mirror", "Quadratic", True if k==0 else False)
        mirror_algorithm.reInitialize(image_loader)
        mirror_algorithm.fibonacci_ratio_search(precision_fibonacci, max_points_fibonacci, cost_tolerance_fibonacci)
        to_result_dict(individual_image_results, image_names, rotation_algorithm, "Mirror", "Fibonacci", True if k==0 else False)
        k+=1


    print("3. Raw Results Computed!\n")

    # 7. PROCESS FINAL RESULTS ##########################################
    def angle_to_pi_pi( angle): # convert any angle to range ()-pi,pi]
        angle= angle%(2*np.pi) # take it to [-2pi, 2pi]
        return angle-np.sign(angle)*2*np.pi if abs(angle)>np.pi else angle

    def num_of_zeros(n): # To count the number of zero decimals before non-zeros
        s = '{:.16f}'.format(n).split('.')[1]
        return len(s) - len(s.lstrip('0'))
    cwd = os.getcwd() # working directory
    '''
    def make_hyperlink(image_path, is_image_average): # for inserting direct link to image paths
        if is_image_average:
            url= f"./SIMULATIONS/nx_{image_name.split('nx_')[-1]}/WHITE_NOISES/AVERAGES/{image_name}/PROFILES_{image_name}.png"
        else:
            url = f"./SIMULATIONS/nx_{image_name.split('nx_')[-1]}/WHITE_NOISES/sigma_{image_name.split('sigma_')[-1]}/SATURATION/satur_{image_name.split('satur_')[-1]}/interpol_{image_name.split('interpol_')[-1]}/PROFILES_interpol_{image_name.split('interpol_')[-1]}.png"
        return f"=HYPERLINK(\"{url}\", \"link\")"
    '''
    conv=1 if deg_or_rad=="rad" else 180/np.pi # conversion factor


    final_results = { 'ground_truth':[], 'averaged_images_or_angles':[], 'min_abs_theoretical_error':[], 'best_correct_decimals':[], 'best_algorithm':[], 'mirror_fibo':[], 'mirror_quad':[], 'rotation_fibo':[], 'rotation_quad':[] }

    # def to_final_dict_results(result_dict, raw_result_dict):
    # we will use a pandas dataframe for it is easier to manipulate
    raw_results = pd.DataFrame.from_dict(individual_image_results)

    # what I will do here is veery inefficient, but well, f* that xD
    pd.options.display.max_colwidth = 100
    for ref_group_tuple, reference_df in raw_results[raw_results["is_reference"]==True].groupby([ 'averaged_before_or_after']):
        # we will avoid experiments crossing sigma, saturation, interpolation and average before or after
        for prob_group_tuple, problem_df in raw_results[(raw_results["is_reference"]==False) & (raw_results["averaged_before_or_after"]==ref_group_tuple[0])].groupby(['averaged_before_or_after']):
            # the only difference between the prob_group_tuple and the ref one will be the phiCR -> the only degree we allow to be crossed in the experiments is that one.
            # if we are interested in some other crossing, just erase the restriction -> And do not forget to put it in the output table as a separate column per ref or pb
            if ref_group_tuple[0]=='B': # averaged_before_or_after==B -> talking about the average images
                assert(reference_df.shape[0]==4)
                assert(problem_df.shape[0]==4)
            else:
                assert(reference_df.shape[0]==4*len(ref_images))
                assert(problem_df.shape[0]==4*len(pb_images))


            if ref_group_tuple[0]=='B':
                final_results["averaged_images_or_angles"].append("images") # before is image, after is angles
                final_results["mirror_fibo"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Mirror") & (problem_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Mirror") & (reference_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0])*conv*polCR)
                final_results["mirror_quad"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Mirror") & (problem_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Mirror") & (reference_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0])*conv*polCR)
                final_results["rotation_fibo"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Rotation") & (problem_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Rotation") & (reference_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].iloc[0])*conv*polCR)
                final_results["rotation_quad"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Rotation") & (problem_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0]-reference_df[(reference_df["polarization_method"]=="Rotation") & (reference_df["optimization_1d"]=="Quadratic")]["found_phiCR"].iloc[0])*conv*polCR)


            else:
                final_results["averaged_images_or_angles"].append("angles") # before is image, after is angles

                final_results["mirror_fibo"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Mirror") & (problem_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].mean()-reference_df[(reference_df["polarization_method"]=="Mirror") & (reference_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].mean())*conv*polCR)
                final_results["mirror_quad"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Mirror") & (problem_df["optimization_1d"]=="Quadratic")]["found_phiCR"].mean()-reference_df[(reference_df["polarization_method"]=="Mirror") & (reference_df["optimization_1d"]=="Quadratic")]["found_phiCR"].mean())*conv*polCR)
                final_results["rotation_fibo"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Rotation") & (problem_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].mean()-reference_df[(reference_df["polarization_method"]=="Rotation") & (reference_df["optimization_1d"]=="Fibonacci")]["found_phiCR"].mean())*conv*polCR)
                final_results["rotation_quad"].append(angle_to_pi_pi(problem_df[(problem_df["polarization_method"]=="Rotation") & (problem_df["optimization_1d"]=="Quadratic")]["found_phiCR"].mean()-reference_df[(reference_df["polarization_method"]=="Rotation") & (reference_df["optimization_1d"]=="Quadratic")]["found_phiCR"].mean())*conv*polCR)


            ground_truth_relative_angle=ground_truth if deg_or_rad=='deg' else angle_to_pi_pi(ground_truth)
            final_results['ground_truth'].append(ground_truth_relative_angle)
            theoretical_errors=np.abs(np.array([ final_results["mirror_fibo"][-1], final_results["mirror_quad"][-1],  final_results["rotation_fibo"][-1], final_results["rotation_quad"][-1]])-ground_truth_relative_angle )
            final_results["min_abs_theoretical_error"].append( theoretical_errors.min()  )
            final_results['best_correct_decimals'].append( num_of_zeros(theoretical_errors.min()) )
            final_results["best_algorithm"].append( ["mirror_fib", "mirror_quad", "rotation_fib", "rotation_quad"][theoretical_errors.argmin()] )
            #final_results["ref_im_link"].append(make_hyperlink(reference_df["Image_Name"].iloc[0], is_image_average=(ref_group_tuple[5]=='B')))
            #final_results["prob_im_link"].append(make_hyperlink(problem_df["Image_Name"].iloc[0], is_image_average=(ref_group_tuple[5]=='B')))

    print("4. Final Results Computed!\n")

    # 8. OUTPUT RESULTS INTO AN EXCEL WITH IMAGES (GIFS) ################
    from styleframe import StyleFrame

    final_results_df = pd.DataFrame.from_dict(final_results)

    writer = StyleFrame.ExcelWriter(f"{output_path}/EXCEL_FINAL_RESULTS_{experiment_name}.xlsx")

    # Convert the dataframe to an XlsxWriter Excel object.
    StyleFrame.A_FACTOR=10
    StyleFrame.P_FACTOR=0.9
    StyleFrame(final_results_df.sort_values(by=['min_abs_theoretical_error'])).set_row_height(1,50).to_excel(writer, sheet_name='Experiment Results by Min Error', index=False, best_fit=list(final_results_df.columns), float_format="%.12f")

    StyleFrame(raw_results).set_row_height(1,50).to_excel(writer,  sheet_name='Raw Results per Image', best_fit=list(raw_results.columns), index=False,  float_format="%.12f")
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    os.remove(f"{output_path}/Background.png")

if __name__ == '__main__':
    import sys
    
    # total arguments
    n = len(sys.argv)
    #print(sys.argv)
    print("Total arguments passed:", n)
    
    if n==0:
        saturation=0.1
        for choice, gt in zip(["ref_vs_ref","sin_el_negativo", "sin_el_positivo", "con_los_dos", "ortog", "43_44", "70_71", "28_29", "17_18", "18_19", "non_noisy_5_6", "non_noisy_72_73"],[0, -13.85, 9.45, -4.4, 90,(2.6544740200042725+1.57120680809021)*180/np.pi/2, (-0.6731816530227661+2.4470927715301514)*180/np.pi/2,(0.6789670586585999-0.9714600443840027)*180/np.pi/2, (0.659442126750946+2.2813968658447266)*180/np.pi/2, (-2.2813968658447266+2.679948091506958)*180/np.pi/2, (-2.6049387454986572+1.7562638521194458)*180/np.pi/2,(-2.946422576904297-1.33404541015625)*180/np.pi/2]):
            output_path=f"/home/oiangu/Desktop/Conical_Refraction_Polarimeter/LAB/EXPERIMENTAL/BENCHMARK_High_Saturation_0.1/{choice}"
            run_angle_live(output_path, gt, saturation, silhouette=False)
    else: # expected, first output_path then saturation, then silhouette
        for choice, gt in zip(["ref_vs_ref","sin_el_negativo", "sin_el_positivo", "con_los_dos", "ortog", "43_44", "70_71", "28_29", "17_18", "18_19", "non_noisy_5_6", "non_noisy_72_73"],[0, -13.85, 9.45, -4.4, 90,(2.6544740200042725+1.57120680809021)*180/np.pi/2, (-0.6731816530227661+2.4470927715301514)*180/np.pi/2,(0.6789670586585999-0.9714600443840027)*180/np.pi/2, (0.659442126750946+2.2813968658447266)*180/np.pi/2, (-2.2813968658447266+2.679948091506958)*180/np.pi/2, (-2.6049387454986572+1.7562638521194458)*180/np.pi/2,(-2.946422576904297-1.33404541015625)*180/np.pi/2]):
            output_path=f"/home/oiangu/Desktop/Conical_Refraction_Polarimeter/LAB/EXPERIMENTAL/BENCHMARK_High_Saturation_0.1/{choice}"
            run_angle_live(sys.argv[1]+f"/{choice}", gt, float(sys.argv[2]), silhouette=True if sys.argv[3]=="True" else False)
