import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
import os
from SOURCE.Ad_Hoc_Optimizer import Ad_Hoc_Optimizer


class Radial_Histogram_Algorithm:
    def __init__(self, image_loader, use_exact_gravicenter):
        self.images = image_loader.centered_ring_images
        self.image_names = image_loader.raw_images_names
        self.mode = image_loader.mode
        self.use_exact_gravicenter=use_exact_gravicenter
        if use_exact_gravicenter:
            self.grav = image_loader.g_centered.squeeze() # squeeze for the case ther is only one im
        else: # then use the image center as radial histogram origin
            self.grav = np.array([self.mode]*2)+0.5
        self.min_angle=0
        self.max_angle=2*np.pi
        self.optimals={}
        self.times={}

    def compute_histogram_binning(self, angle_bin_size):
        if not self.use_exact_gravicenter or len(self.grav.shape)==1:
            t=time()
            cols = np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1) ) #[h,w]
            rows = cols.swapaxes(0,1) #[h,w]
            # angles relative to the  gravicenter in range [-pi,pi] but for the reversed image
            index_angles = np.arctan2( rows-self.grav[0], cols-self.grav[1] ) #[h,w]
            # unfortunately in order for the binning to work we will need to have it in 0,2pi (else the 0 bin gets too big)
            # we take advantage of this to already left the angles inverted too
            index_angles[index_angles>0] = 2*np.pi-index_angles[index_angles>0]
            index_angles[index_angles<0] *= -1

            bins = (index_angles//angle_bin_size).astype(int)
            # assign angles to bins and sum intensities of pixels in the same bins
            histograms=np.array([np.bincount( bins.flatten(), weights=im.flatten() ) for im in self.images])
            t= self._round_to_sig((time()-t)/histograms.shape[0])
            for name in self.image_names:
                self.times[name] = t

        else:
            cols = np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1) ) #[h,w]
            rows = cols.swapaxes(0,1) #[h,w]
            histograms=[]
            for grav, im, name in zip(self.grav, self.images, self.image_names):
                t=time()
                # angles relative to the  gravicenter in range [-pi,pi] but for the reversed image
                index_angles = np.arctan2( rows-grav[0], cols-grav[1] ) #[h,w]
                # unfortunately in order for the binning to work we will need to have it in 0,2pi (else the 0 bin gets too big)
                # we take advantage of this to already left the angles inverted too
                index_angles[index_angles>0] = 2*np.pi-index_angles[index_angles>0]
                index_angles[index_angles<0] *= -1

                bins = (index_angles//angle_bin_size).astype(int)
                # assign angles to bins and sum intensities of pixels in the same bins
                histograms.append(np.bincount( bins.flatten(), weights=im.flatten() ) )
                t=time()-t
                self.times[name]=self._round_to_sig(t)
            histograms=np.array(histograms)

        self.histograms=histograms
        self.precisions = self._round_to_sig(angle_bin_size/2.0)
        self.angles=np.arange(start=self.min_angle, stop=self.max_angle, step=angle_bin_size, dtype=np.float64)[:histograms.shape[1]]+angle_bin_size/2
        for image_name, histogram in zip(self.image_names, histograms):
            # until no stauration images re obtained this is the way (look for the minimum instead of maximum!)
            self.optimals[image_name] = self._round_to_sig(self.angles[np.argmin(histogram)]-np.pi, self.precisions)

    def compute_histogram_masking(self, angle_bin_size):
        # if use img center or only one image
        if not self.use_exact_gravicenter or len(self.grav.shape)==1:
            t=time()
            # create an array with the column number at each element and one with row numbers
            cols = np.expand_dims(np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)),2) #[h,w, 1]
            rows = cols.swapaxes(0,1) #[h,w, 1]
            # note that we set a minus sign for the angles in order to account for the pixel coordinate system and agree with the rest of algorithms (but in reality wrt the image the angles representing them are the same but *-1)
            angles = -np.flip(np.arange(start=self.min_angle, stop=self.max_angle, step=angle_bin_size, dtype=np.float64)) #[N_theta]
            # create masks for each half plane at different orientations
            greater=np.greater(rows, np.tan(angles)*(cols-self.grav[0])+self.grav[1]) #[h,w,N_theta]
            #smaller=np.logical_not(greater) #[h,w,N_theta]
            # for angles in [-2pi,-3pi/2] and [-pi/2,0] the mask should be true if col greater than smallest angle and col smaller than greatest angle of bin
            # for angles in [-3pi/2, -pi/2] the mask should be true if smaller than smallest angle and greater than greatest angle of bin
            bin_lower = np.concatenate((greater[:,:,angles<-3*np.pi/2], np.logical_not(greater)[:,:,(angles>-3*np.pi/2)&(angles<-np.pi/2)], greater[:,:,angles>-np.pi/2]), axis=2) #[h,w,N_theta]
            #bin_higher = np.logical_not(bin_lower) #[h,w,N_theta]
            # get the pizza like masks. We have one mask per bin (N_bins=N_theta-1)
            masks=np.logical_and(bin_lower[:,:,:-1], np.logical_not(bin_lower)[:,:,1:]) # [h,w,N_theta-1]

            # Prepare the images for being masked for each bin
            # [N_images, h, w]-> [N_theta-1, N_images, h, w]->[N_images, h, w, N_theta-1]
            #images = np.moveaxis( np.broadcast_to(self.images,
            #                               (angles.shape[0]-1)+self.images.shape), 0,-1)
            # apparently there is no way to broadcast correctly the mask preserving that dimension, so will need a comprehension instead
            histograms=np.array([np.sum(self.images[:, masks[:,:,j]], axis=1) for j in range(angles.shape[0]-1)]).swapaxes(0,-1) #[N_images, N_theta-1] Intensities per bin
            t= self._round_to_sig((time()-t)/histograms.shape[0])
            for name in self.image_names:
                self.times[name] = t
        else: # then each image has its own masks (they will be slightly different)
            # create an array with the column number at each element and one with row numbers
            cols = np.expand_dims(np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)),2) #[h,w, 1]
            rows = cols.swapaxes(0,1) #[h,w, 1]
            # note that we set a minus sign for the angles in order to account for the pixel coordinate system and agree with the rest of algorithms (but in reality wrt the image the angles representing them are the same but *-1)
            angles = -np.flip(np.arange(start=self.min_angle, stop=self.max_angle, step=angle_bin_size, dtype=np.float64)) #[N_theta]
            histograms=[]
            for grav, im, name in zip(self.grav, self.images, self.image_names):
                t=time()
                # create masks for each half plane at different orientations
                greater=np.greater(rows, np.tan(angles)*(cols-grav[0])+grav[1]) #[h,w,N_theta]
                #smaller=np.logical_not(greater) #[h,w,N_theta]
                # for angles in [-2pi,-3pi/2] and [-pi/2,0] the mask should be true if col greater than smallest angle and col smaller than greatest angle of bin
                # for angles in [-3pi/2, -pi/2] the mask should be true if smaller than smallest angle and greater than greatest angle of bin
                bin_lower = np.concatenate((greater[:,:,angles<-3*np.pi/2], np.logical_not(greater)[:,:,(angles>-3*np.pi/2)&(angles<-np.pi/2)], greater[:,:,angles>-np.pi/2]), axis=2) #[h,w,N_theta]
                #bin_higher = np.logical_not(bin_lower) #[h,w,N_theta]
                # get the pizza like masks. We have one mask per bin (N_bins=N_theta-1)
                masks=np.logical_and(bin_lower[:,:,:-1], np.logical_not(bin_lower)[:,:,1:]) # [h,w,N_theta-1]
                histograms.append(np.array([np.sum(im[masks[:,:,j]]) for j in range(angles.shape[0]-1)])) #[N_theta-1] Intensities per bin
                t=time()-t
                self.times[name]=self._round_to_sig(t)
            histograms=np.array(histograms)

        self.histograms=histograms
        self.precisions = self._round_to_sig(angle_bin_size/2)
        self.angles=-(angles+self.precisions)[:-1] # centers
        for image_name, histogram in zip(self.image_names, histograms):
            self.optimals[image_name] = self._round_to_sig(-(angles[np.argmin(histogram)]+angle_bin_size/2)-np.pi, self.precisions)


    def compute_histogram_interpolate(self, angle_bin_size):
        pass

    def refine_by_cosine_fit(self):
        pass

    def save_result_plots(self, output_path, title, cosine_fit=None):
        "Maybe add the option or check of whether cosine fit should also be plotted"
        os.makedirs(f"{output_path}/Histogram_Algorithm/", exist_ok=True)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        tag="Exact_Grav" if self.use_exact_gravicenter else "Image_Center"
        for name, bin_sums in zip(self.image_names, self.histograms):
            ax.bar(self.angles, bin_sums, align='center', width=2*self.precisions, label=name)
            ax.set_title(f"Optimal angle={self.optimals[name]}+-{self.precisions} rad\nComputed Bins={bin_sums.shape[0]} . Eff.Time Required={self.times[name]}s")
            ax.set_xlabel("Angle (rad)")
            ax.set_ylabel("Total Intensity")
            #ax.set_yscale('log')
            ax.grid(True)
            plt.savefig(f"{output_path}/Histogram_Algorithm/{title}_{tag}_{name}.png")
            ax.clear()

    def _round_to_sig(self, x_to_round, reference=None, sig=2):
        reference = x_to_round if reference is None else reference
        return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)

class Mirror_Flip_Algorithm:
    def __init__(self, image_loader, min_angle, max_angle, interpolation_flag, initial_guess_delta, method use_exact_gravicenter):
        """
            Argument image_loader is expected to be an instance of class Image_Loader,
            which has already been initialized and has non-null attributes:
            - self.mode: 203 or 607 depending on whther it contians i607 or i203 images
            - self.centered_ring_images: the [N_images, self.mode*2+1 (h), self.mode*2+1 (w)]
            - self.g_centered: intensity gravicenter in pixel index coordinates [N_images, 2 (h,w)]
            - self.raw_images_names: nems of the N_images in order

        - initial_guess_angle_delta (float): The two initial point ssampled in the middle of the
            initial interval should be almost exactly in the middle, but still slightly separted
            netween them. This will maximize the interval reduction. The present parameter
            will measure their distance.

        """
        self.original_images = image_loader
        self.interpolation_flag = interpolation_flag
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.initial_guess_delta = initial_guess_delta
        self.mode = image_loader.mode
        self.cols=None
        self.index_angles=None
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.precisions={}
        self.times={}
        self.method=method
        if method=="bin":
            self.optimizer = Ad_Hoc_Optimizer(min_angle, max_angle, initial_guess_delta, self.evaluate_mirror_bin)
        elif method=="mask":
            self.optimizer = Ad_Hoc_Optimizer(min_angle, max_angle, initial_guess_delta, self.evaluate_mirror_mask)
        else: # interpolating method
            self.optimizer = Ad_Hoc_Optimizer(min_angle, max_angle, initial_guess_delta, self.evaluate_mirror_affine)

        if left_vs_right:
            self.udlr=1 # minimize cost
        else: #up vs down
            self.udlr=-1 # maximize cost

        self.use_exact_gravicenter=use_exact_gravicenter
        if use_exact_gravicenter:
            self.grav=self.original_images.g_centered #[N_images, 2(h,w)]
        else:
            # gravicenter the same for all
            self.grav=np.array(2*[self.mode])+0.5


    def mirror_flip_at_angle(self, image_array, angle, center):
        """
        Center is expected to be a point [h,w]
        """
        a=np.cos(2*angle)
        b=np.sin(2*angle)
        mirror_mat=np.float32([[a, b, center[1]*(1-a)-center[0]*b],
                             [b, -a, center[0]*(1+a)-center[1]*b]])
        return cv2.warpAffine(image_array, mirror_mat, image_array.shape, flags=self.interpolation_flag).astype(image_array.dtype)

    def evaluate_mirror_affine(self, image_array, angle, center, udlr=1):
        # if udlr is 1 then we are minimizing the difference -> LR mode
        # if udlr is -1 then we are maximizing the difference -> UD mode
        return udlr*np.sum(np.abs(self.mirror_flip_at_angle(image_array, angle, center)-image_array))


    def evaluate_mirror_bin(self, image, angle, angles_to_grav, udlr=1):
        # angles in [-pi/2, pi/2] are fine
        mask=(angles_to_grav>angle) & (angles_to_grav<(angle+np.pi)) #[h,w]
        return udlr*np.abs(np.sum(image[mask])-np.sum(image[np.logical_not(mask)]))

    def evaluate_mirror_mask(self, image, angle, rows, cols, center, udlr=1):
        # angles must be in [-pi/2,pi/2] or [3pi/2,2pi]
        mask=np.greater(rows, np.tan(angle)*(cols-center[0])+center[1]) #[h,w]
        return udlr*np.abs(np.sum(image[mask])-np.sum(image[np.logical_not(mask)]))

    def prepare_arguments(self, im):
        if self.method="mask":
            # avoid this being generated for every image
            self.cols = self.cols if self.cols is not None else np.expand_dims(np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)),2) #[h,w, 1]
            return (self.cols, self.cols.swapaxes(0,1),
                self.grav[im] if self.use_exact_gravicenter else self.grav, self.udlr)

        elif self.method="bin":
            self.cols = self.cols if self.cols is not None else np.expand_dims(np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)),2) #[h,w, 1]
            # angles relative to the  gravicenter in range [-pi,pi] but for the reversed image
            if self.use_exact_gravicenter:
                index_angles = -np.arctan2( self.cols.swapaxes(0,1)-grav[im][0], cols-grav[im][1] ) #[h,w]
            else:
                self.index_angles = self.index_angles if self.index_angles is not None else -np.arctan2( self.cols.swapaxes(0,1)-grav[0], cols-grav[1] ) #[h,w]
            return ( index_angles,self.udlr)
        else: # affine
            return (self.grav[im] if self.use_exact_gravicenter else self.grav, self.udlr)

    def given_axis_angle_subtract_L_to_R(self):
        # such that if the output is positive, then R has more intensity and you know immediately that the good angle is the bigger one?
        # de fet esto sugiere un algoritmo con el polano ortogonal que directamente te encuentra el angulo que toca, pero bueno con los que buscan el eje simetrico el truco no parece que funcionara

    def get_polarization_angle(self, angle, image):
        """
        All the mirror methods have the problem that we only get the
        correct angle up to an angle pi. In order to know which is the
        angle to the maximum of the ring (and not the minimum) a final
        subtle check is required.
        """
        if self.udlr==-1: # then what we have is the orthogonal plane to the symmetry axis
            angle+=np.pi/2 # so we translate it into the symetry axis



    def brute_force_search(self, angle_steps, zoom_ratios):
        """
        What does this exactly do

        Arguments
        --------
        - angle_steps (list): A list of the different angle steps to take in each of the sweeps.
            Expected N, where N is the number of sweeps that will be performed. The first one is
            expected to be the coarsest grain and they should be ordered from big to smallest.
            The last step in the list will define the precision of the found minimum. The angle
            steps are expected to be in (0, 2pi)

        - zoom_ratios (list): A list of the interval reductions that will be held after each sweep
            around the current best candidate for the minimum. There should be N-1 elements and
            they should be numbers in (0,1].

        """
        zoom_ratios.append(1) #to avoid out of index in the last iteration
        for im, image_name in enumerate(self.original_images.raw_images_names):
            (self.times[f"Brute_Force_{image_name}"],
            self.computed_points[f"Brute_Force_{image_name}"],
            self.optimals[f"Brute_Force_{image_name}"],
            self.optimums[f"Brute_Force_{image_name}"],
            self.precisions[f"Brute_Force_{image_name}"]) = self.optimizer.brute_force_search(
                    angle_steps, zoom_ratios,
                    self.original_images.centered_ring_images[im], self.prepare_arguments(im))


    def fibonacci_ratio_search(self, precision, maximum_points, cost_tol):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            rotation arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision will
            be outputed accordingly.

        - maximum_points (int): Maximum number of points to use in the minimum search. It is also
            the number of times to make an interval reduction.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.
        """

        for im, image_name in enumerate(self.original_images.raw_images_names):
            (self.times[f"Fibonacci_Search_{image_name}"],
            self.computed_points[f"Fibonacci_Search_{image_name}"],
            self.optimals[f"Fibonacci_Search_{image_name}"],
            self.optimums[f"Fibonacci_Search_{image_name}"],
            self.precisions[f"Fibonacci_Search_{image_name}"])=                self.optimizer.fibonacci_ratio_search(
                    precision, maximum_points, cost_tol,
                    self.original_images.centered_ring_images[im], self.prepare_arguments(im))



    def quadratic_fit_search(self, precision, max_iterations, cost_tol):
        """
        Quadratic

        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            rotation arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision will
            be outputed accordingly.

        - max_iterations (int): Number of maximum iterations of quadratic function fit and
            minimization to tolerate.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        """
        for im, image_name in enumerate(self.original_images.raw_images_names):
            (self.times[f"Quadratic_Search_{image_name}"],
            self.computed_points[f"Quadratic_Search_{image_name}"],
            self.optimals[f"Quadratic_Search_{image_name}"],
            self.optimums[f"Quadratic_Search_{image_name}"],
            self.precisions[f"Quadratic_Search_{image_name}"])=self.optimizer.quadratic_fit_search(
                precision, max_iterations, cost_tol,
                self.original_images.centered_ring_images[im], self.prepare_arguments(im))


class Gradient_Algorithm:
    def __init__(self, image_loader, min_radious, max_radious, initial_guess_delta, use_exact_gravicenter):
        self.optimizer = Ad_Hoc_Optimizer(min_radious, max_radious, initial_guess_delta, self.evaluate_mask_radious)
        self.original_images = image_loader
        self.use_exact_gravicenter=use_exact_gravicenter
        #self.save_images(self.mirror_images_wrt_width_axis, "./OUTPUT/", [name+"_mirror" for name in self.original_images.raw_images_names])
        self.min_radious = min_radious
        self.max_radious = max_radious
        self.initial_guess_delta = initial_guess_delta
        self.mode = image_loader.mode
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.precisions={}
        self.times={}
        self.angles={}
        self.masked_gravs={}

        if use_exact_gravicenter: #[N_images, 2 (h,w)]
            self.grav = image_loader.g_centered.squeeze() # squeeze in case there is only one image
        else: # then use the image center as gravicenter
            self.grav = np.array([self.mode]*2)+0.5

        # compute the distance matrices to the gravicenter of the images
        cols=np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
        rows=cols.swapaxes(0,1) #[h,w]
        if not self.use_exact_gravicenter or len(self.grav.shape)==1: #[h,w]
            # all the images have the same mask for distance to center
            self.distances_to_grav = (cols-self.grav[0])**2+(rows-self.grav[1])**2
        else: # [N_images, h,w]
            self.distances_to_grav = np.array([(cols-grav[0])**2+(rows-grav[1])**2
                for grav in self.grav])

    def compute_new_gravicenter(self, image, radious, distances_to_grav):
        circle=np.where(distances_to_grav<=radious**2, image, 0) #[h,w]
        # compute the gravicenter of the masked image
        intensity_in_w = np.sum(circle, axis=0) # weights for x [w]
        intensity_in_h = np.sum(circle, axis=1) # weights for y [h]
        total_intensity = intensity_in_h.sum()
        return [np.dot(intensity_in_h, np.arange(circle.shape[0]))/total_intensity,
         np.dot(intensity_in_w, np.arange(circle.shape[1]))/total_intensity]

    def save_images(self, images, output_path, names):
        if type(names) is not list:
            images=[images,]
            names = [names,]
        for name, image in zip(names, images):
            cv2.imwrite(f"{output_path}/{name}.png", image)

    def evaluate_mask_radious(self, image, radious, distances_to_grav, grav):
        # mask the image in the circumference
        circle=np.where(distances_to_grav<=radious**2, image, 0) #[h,w]
        #self.save_images(circle.astype(np.uint8), '.', f"Rad={radious}")
        # compute the gravicenter of the masked image
        intensity_in_w = np.sum(circle, axis=0) # weights for x [w]
        intensity_in_h = np.sum(circle, axis=1) # weights for y [h]
        total_intensity = intensity_in_h.sum()
        new_grav = [np.dot(intensity_in_h, np.arange(circle.shape[0]))/total_intensity,
            np.dot(intensity_in_w, np.arange(circle.shape[1]))/total_intensity]
        return -((new_grav[1]-grav[1])**2+(new_grav[0]-grav[0])**2)
        # the minus sign is because the algorithms will try to minimize the cost (and here we
        # are looking for the maximum)

    def brute_force_search(self, radii_steps, zoom_ratios):
        zoom_ratios.append(1) #to avoid out of index in the last iteration
        mul=False if len(self.distances_to_grav.shape)==2 else True
        for im, image_name in enumerate(self.original_images.raw_images_names):
            name=f"Brute_Force_{image_name}"
            grav= self.grav[im] if mul else self.grav
            (self.times[name],
            self.computed_points[name],
            self.optimals[name],
            self.optimums[name],
            self.precisions[name]) = self.optimizer.brute_force_search(
                radii_steps, zoom_ratios, self.original_images.centered_ring_images[im],
                (self.distances_to_grav[im] if mul else self.distances_to_grav,
                 grav))
            # Now that the optimal Radious is set, we compute the polarization angle
            masked_grav=self.compute_new_gravicenter(
                self.original_images.centered_ring_images[im],
                self.optimals[name][f"Stage_{len(radii_steps)-1}"],
                self.distances_to_grav[im] if mul else self.distances_to_grav)
            self.angles[name]=-np.arctan2(masked_grav[0]-grav[0], masked_grav[1]-grav[1])
            self.masked_gravs[name]=masked_grav


    def fibonacci_ratio_search(self, precision, maximum_points, cost_tol):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            rotation arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision will
            be outputed accordingly.

        - maximum_points (int): Maximum number of points to use in the minimum search. It is also
            the number of times to make an interval reduction.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.
        """

        mul=False if len(self.distances_to_grav.shape)==2 else True
        for im, image_name in enumerate(self.original_images.raw_images_names):
            name = f"Fibonacci_Search_{image_name}"
            grav= self.grav[im] if mul else self.grav
            (self.times[name],
            self.computed_points[name],
            self.optimals[name],
            self.optimums[name],
            self.precisions[name])=                self.optimizer.fibonacci_ratio_search(
                    precision, maximum_points, cost_tol,
                     self.original_images.centered_ring_images[im],
                    (self.distances_to_grav[im] if mul else self.distances_to_grav,
                     grav))
            # Now that the optimal Radious is set, we compute the polarization angle
            masked_grav=self.compute_new_gravicenter(
                self.original_images.centered_ring_images[im],
                self.optimals[name],
                self.distances_to_grav[im] if mul else self.distances_to_grav)
            self.angles[name]=-np.arctan2(masked_grav[0]-grav[0], masked_grav[1]-grav[1])
            self.masked_gravs[name]=masked_grav



    def quadratic_fit_search(self, precision, max_iterations, cost_tol):
        """
        Quadratic

        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            rotation arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision will
            be outputed accordingly.

        - max_iterations (int): Number of maximum iterations of quadratic function fit and
            minimization to tolerate.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        """
        mul=False if len(self.distances_to_grav.shape)==2 else True
        for im, image_name in enumerate(self.original_images.raw_images_names):
            name=f"Quadratic_Search_{image_name}"
            grav= self.grav[im] if mul else self.grav
            (self.times[name],
            self.computed_points[name],
            self.optimals[name],
            self.optimums[name],
            self.precisions[name])=self.optimizer.quadratic_fit_search(
                precision, max_iterations, cost_tol,
                self.original_images.centered_ring_images[im],
               (self.distances_to_grav[im] if mul else self.distances_to_grav,
                grav))
            masked_grav=self.compute_new_gravicenter(
                self.original_images.centered_ring_images[im],
                self.optimals[name],
                self.distances_to_grav[im] if mul else self.distances_to_grav)
            self.angles[name]=-np.arctan2(masked_grav[0]-grav[0], masked_grav[1]-grav[1])
            self.masked_gravs[name]=masked_grav


    def plot_gravicenter_mask_diagram(self, image, grav, masked_grav, distances_to_grav, radious, ax, title):
        ax.clear()
        ax.imshow(image, label="Image")
        ax.set_ylabel("height")
        ax.set_xlabel("width")
        ax.set_title(title)
        ax.plot(grav[1], grav[0], 'or', markersize=4, label="Full image gravicenter")
        ax.plot(masked_grav[1], masked_grav[0], 'ow', markersize=4, label="Masked circle gravicenter")
        ax.imshow(distances_to_grav>radious**2, alpha=0.5, label="Optimal Radious Mask")
        ax.legend()


    def save_result_plots_brute_force(self, output_path):
        os.makedirs(f"{output_path}/Gradient_Algorithm/", exist_ok=True)
        fig, axes = plt.subplots(len(next(iter(self.times.values())).values())+1, 1, figsize=(10,15))
        fig.tight_layout(pad=5.0)
        mul=False if len(self.distances_to_grav.shape)==2 else True # if using exact gravcter
        for im, (name, computed_points) in enumerate(self.computed_points.items()):
            for stage, ax in enumerate(axes[:-1]):
                ax.clear()
                ax.plot(computed_points[f"Stage_{stage}"][:,0],
                    computed_points[f"Stage_{stage}"][:,1], 'o', markersize=2,
                    label=f"{name}_Stage_{stage}")
                ax.set_title(f"STAGE {stage}: Optimal radoius={self.optimals[name][f'Stage_{stage}']}+-{self.precisions[name][f'Stage_{stage}']} rad\nComputed Points={len(computed_points[f'Stage_{stage}'])}. Time Required={self.times[name][f'Stage_{stage}']}s")
                ax.set_xlabel("Radious (pixels)")
                ax.set_ylabel("-||Masked Gravicenter-Full Gravicenter||^2")
                #ax.set_yscale('log')
                ax.grid(True)
            self.plot_gravicenter_mask_diagram(self.original_images.centered_ring_images[im],
                self.grav[im] if mul else self.grav, self.masked_gravs[name],
                self.distances_to_grav[im] if mul else self.distances_to_grav,
                self.optimals[name][f'Stage_{len(axes)-2}'], axes[-1],
                f"Polarization Angle {self.angles[name]} rad" )
            plt.savefig(f"{output_path}/Gradient_Algorithm/{name}.png")

    def save_result_plots_fibonacci_or_quadratic(self, output_path):
        os.makedirs(f"{output_path}/Gradient_Algorithm/", exist_ok=True)
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        mul=False if len(self.distances_to_grav.shape)==2 else True # if using exact gravcter
        for im, (name, computed_points) in enumerate(self.computed_points.items()):
            ax1.plot(computed_points[:,0], computed_points[:,1], 'o', label=name)
            ax1.set_title(f"Optimal Radious={self.optimals[name]}+-{self.precisions[name]} rad\nComputed Points={len(computed_points)} . Time Required={self.times[name]}s")
            ax1.set_xlabel("Radious (pixels)")
            ax1.set_ylabel("-||Masked Gravicenter-Full Gravicenter||^2")
            #ax.set_yscale('log')
            ax1.grid(True)
            self.plot_gravicenter_mask_diagram(self.original_images.centered_ring_images[im],
                self.grav[im] if mul else self.grav, self.masked_gravs[name],
                self.distances_to_grav[im] if mul else self.distances_to_grav,
                self.optimals[name], ax2,
                f"Polarization Angle {self.angles[name]} rad" )
            plt.savefig(f"{output_path}/Gradient_Algorithm/{name}.png")
            ax1.clear()



class Rotation_Algorithm:
    """
    The distance between the images, i607 and rotated i607R as a function of the rotation angle,
    once both images are centered in the gravicenter, is a strictly unimodal and convex function,
    the minimum of which should be extremely simple to find (disregarding the fluctuations that
    might appear in the microscopic rotation scale due to the discrete nature of images or the
    lossy compression issues). A simple algorithm that can prove to be good enough is the Fibonacci
    Ratio three point search.

    Brute force grid search:
    -   Hace run grid de step tal grosso, buscar los 3 ptos minimos, coger el bounding box y hacer
        un grid mas fino con tantos puntos como antes. Y otra vez. Aka haciendo zoom ins. Asi hasta
        que la distancia en metrica entre los tres puntos sea menor a una tolerancia epsilon. En
        cada iteracion podemos plotear el grafico.

    - Golden ratio approach. Tope simple y rapido. Guardo los puntos que salgan para un plot final.

    - Y yet another que es even better existe casi seguro. Algo con una componente mas estocastica.

    OJOOOO!!!!
    - Si realizo el mirror flip respecto al gravicentro exacto de cada imagen en en eje y?
    - Y si realizo las rotaciones de la imagen relativas al gravicentro exacto de la imagen???
    Si hago estas dos correciones, incluso accounteara por esas decimas problematicas!!

    Otro suggetsion de algoritmo! Podrias ir probando mirror reflections relativos a angulos
    diferentes a rectas ke pasan desde el gravicenter hasta encontrar el mirror reflection tal que
    divide en una mitad y en la otra la suma de intensidades mas parecida entre ellas. Super
    facil de hacerlo y se pueden usar los mismos tres algoritmos de busqueda si cambias la
    funcion de coste
    """
    def __init__(self, image_loader, min_angle, max_angle, interpolation_flag, initial_guess_delta, use_exact_gravicenter):
        """
            Argument image_loader is expected to be an instance of class Image_Loader,
            which has already been initialized and has non-null attributes:
            - self.mode: 203 or 607 depending on whther it contians i607 or i203 images
            - self.centered_ring_images: the [N_images, self.mode*2+1 (h), self.mode*2+1 (w)]
            - self.g_centered: intensity gravicenter in pixel index coordinates [N_images, 2 (h,w)]
            - self.raw_images_names: nems of the N_images in order

        - initial_guess_angle_delta (float): The two initial point ssampled in the middle of the
            initial interval should be almost exactly in the middle, but still slightly separted
            netween them. This will maximize the interval reduction. The present parameter
            will measure their distance.

        """
        self.original_images = image_loader
        self.interpolation_flag = interpolation_flag
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.initial_guess_delta = initial_guess_delta
        self.mode = image_loader.mode
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.precisions={}
        self.times={}
        self.optimizer = Ad_Hoc_Optimizer(min_angle, max_angle, initial_guess_delta, self.evaluate_image_rotation)
        self.use_exact_gravicenter=use_exact_gravicenter
        if use_exact_gravicenter:
            self.grav=self.original_images.g_centered #[N_images, 2(h,w)]
            # custom mirror flip the images about the line through gravicenter
            self.mirror_images_wrt_width_axis=np.array([
                self.horizontal_mirror_flip_crossing(
                    self.original_images.centered_ring_images[im], self.grav[im][0])
                for im in range(self.grav.shape[0])]) #[N_images, h,w]
        else:
            # gravicenter the same for all
            self.grav=np.array(2*[self.mode])+0.5
            # mirror flipping image is trivial
            self.mirror_images_wrt_width_axis = np.flip(image_loader.centered_ring_images, 1)
            #self.save_images(self.mirror_images_wrt_width_axis, "./OUTPUT/", [name+"_mirror" for name in self.original_images.raw_images_names])


    def save_images(self, images, output_path, names):
        if type(names) is not list:
            images=[images,]
            names = [names,]
        for name, image in zip(names, images):
            cv2.imwrite(f"{output_path}/{name}.png", image)

    def horizontal_mirror_flip_crossing(self, image_array, h_point):
        # h_point is expected to be point in height such that the flip is made on w
        mirror_mat = np.float32([[1,0,0],[0,-1,2*h_point]]) # apparently 3rd row of affine transformation is redundant for the warAffine function
        return cv2.warpAffine( image_array, mirror_mat, image_array.shape,
                              flags=self.interpolation_flag).astype(image_array.dtype)
    # Perhaps after affine transformation image should be left in float type instead of
    # cropping to int, as it is meant to be used simply for calculations

    def rotate_image_by(self, image_array, angle, center):
        """
        Center is expected to be a point [h,w]
        """
        a=np.cos(angle)
        b=np.sin(angle)
        rot_mat=np.float32([[a, b, center[1]*(1-a)-center[0]*b],
                             [-b, a, center[1]*b+center[0]*(1-a)]])
        return cv2.warpAffine(image_array, rot_mat, image_array.shape, flags=self.interpolation_flag).astype(image_array.dtype)


    def evaluate_image_rotation(self, reference_image, angle, image_to_rotate, center):
        return np.sum(np.abs(self.rotate_image_by(image_to_rotate, angle, center)-reference_image))


    def brute_force_search(self, angle_steps, zoom_ratios):
        """
        What does this exactly do

        Arguments
        --------
        - angle_steps (list): A list of the different angle steps to take in each of the sweeps.
            Expected N, where N is the number of sweeps that will be performed. The first one is
            expected to be the coarsest grain and they should be ordered from big to smallest.
            The last step in the list will define the precision of the found minimum. The angle
            steps are expected to be in (0, 2pi)

        - zoom_ratios (list): A list of the interval reductions that will be held after each sweep
            around the current best candidate for the minimum. There should be N-1 elements and
            they should be numbers in (0,1].

        """
        zoom_ratios.append(1) #to avoid out of index in the last iteration
        for im, image_name in enumerate(self.original_images.raw_images_names):
            (self.times[f"Brute_Force_{image_name}"],
            self.computed_points[f"Brute_Force_{image_name}"],
            self.optimals[f"Brute_Force_{image_name}"],
            self.optimums[f"Brute_Force_{image_name}"],
            self.precisions[f"Brute_Force_{image_name}"]) = self.optimizer.brute_force_search(
                    angle_steps, zoom_ratios,
                    self.original_images.centered_ring_images[im], (self.mirror_images_wrt_width_axis[im],
                    self.grav[im] if self.use_exact_gravicenter else self.grav))


    def fibonacci_ratio_search(self, precision, maximum_points, cost_tol):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            rotation arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision will
            be outputed accordingly.

        - maximum_points (int): Maximum number of points to use in the minimum search. It is also
            the number of times to make an interval reduction.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.
        """

        for im, image_name in enumerate(self.original_images.raw_images_names):
            (self.times[f"Fibonacci_Search_{image_name}"],
            self.computed_points[f"Fibonacci_Search_{image_name}"],
            self.optimals[f"Fibonacci_Search_{image_name}"],
            self.optimums[f"Fibonacci_Search_{image_name}"],
            self.precisions[f"Fibonacci_Search_{image_name}"])=                self.optimizer.fibonacci_ratio_search(
                    precision, maximum_points, cost_tol,
                    self.original_images.centered_ring_images[im], (self.mirror_images_wrt_width_axis[im],
                    self.grav[im] if self.use_exact_gravicenter else self.grav))



    def quadratic_fit_search(self, precision, max_iterations, cost_tol):
        """
        Quadratic

        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            rotation arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision will
            be outputed accordingly.

        - max_iterations (int): Number of maximum iterations of quadratic function fit and
            minimization to tolerate.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        """
        for im, image_name in enumerate(self.original_images.raw_images_names):
            (self.times[f"Quadratic_Search_{image_name}"],
            self.computed_points[f"Quadratic_Search_{image_name}"],
            self.optimals[f"Quadratic_Search_{image_name}"],
            self.optimums[f"Quadratic_Search_{image_name}"],
            self.precisions[f"Quadratic_Search_{image_name}"])=self.optimizer.quadratic_fit_search(
                precision, max_iterations, cost_tol,
                self.original_images.centered_ring_images[im], (self.mirror_images_wrt_width_axis[im],
                    self.grav[im] if self.use_exact_gravicenter else self.grav))


    def save_result_plots_fibonacci_or_quadratic(self, output_path):
        """
        Save the resulting explored points in cost function vs angle, together with the info
        about the optimization.
        """
        os.makedirs(f"{output_path}/Rotation_Algorithm/", exist_ok=True)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        for name, computed_points in self.computed_points.items():
            ax.plot(computed_points[:,0], computed_points[:,1], 'o', label=name)
            ax.set_title(f"Optimal angle={self.optimals[name]}+-{self.precisions[name]} rad\nComputed Points={len(computed_points)} . Time Required={self.times[name]}s")
            ax.set_xlabel("Angle (rad)")
            ax.set_ylabel("|Rotated-Original|")
            #ax.set_yscale('log')
            ax.grid(True)
            plt.savefig(f"{output_path}/Rotation_Algorithm/{name}.png")
            ax.clear()

    def save_result_plots_brute_force(self, output_path):
        os.makedirs(f"{output_path}/Rotation_Algorithm/", exist_ok=True)
        fig, axes = plt.subplots(len(next(iter(self.times.values())).values()), 1, figsize=(10,15))
        fig.tight_layout(pad=5.0)
        for name, computed_points in self.computed_points.items():
            for stage, ax in enumerate(axes):
                ax.clear()
                ax.plot(computed_points[f"Stage_{stage}"][:,0],
                    computed_points[f"Stage_{stage}"][:,1], 'o', markersize=2,
                    label=f"{name}_Stage_{stage}")
                ax.set_title(f"STAGE {stage}: Optimal angle={self.optimals[name][f'Stage_{stage}']}+-{self.precisions[name][f'Stage_{stage}']} rad\nComputed Points={len(computed_points[f'Stage_{stage}'])}. Time Required={self.times[name][f'Stage_{stage}']}s")
                ax.set_xlabel("Angle (rad)")
                ax.set_ylabel("|Rotated-Original|")
                #ax.set_yscale('log')
                ax.grid(True)
            plt.savefig(f"{output_path}/Rotation_Algorithm/{name}.png")
