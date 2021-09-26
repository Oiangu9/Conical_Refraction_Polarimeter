import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
import os
from SOURCE.Ad_Hoc_Optimizer import Ad_Hoc_Optimizer
from SOURCE.Theoretical_Ring_Simulator import *
try:
    from SOURCE.GPU_Classes import *
except:
    pass
import logging

"""
TOOD:

- Hacer que una instancia de algorithm inicializada pueda recibir la misma mas imagenes
osea un init adicional quizas para reiniciar las cosas cada vez que image loader tiene nuevas
imagenes (los dictionary especialmente). Also hacer que las imagenes nuevas sean a las que apunten.

- La clase del camera debe tener el metodo para tomar las reference que es tomar las fotos y
ejecutar el algorithm de busqueda y darle a fix reference.

- La clase camera debe tener el metodo de continous para x frames a tomar de n en n. Se grabean
todas las fotos en un np array y se pasan a image loader que los procesará y llamas al algoritmo
de optimizacion que le puede entrar de argumento admeas de la clase del algoritmo como un lambda
function. asi no hay que hacer ifs dentro del camera si no en el main thread. Cada n fotos tomadas
comprueba que stop no sea true, y sigue hasta hacer todas las fotos o que stop sea true (que lo
tendras que hace run metodo que lo haga true si el usuario clica en stop).

- Also la clase de la camara debera gestionar si se quieren guardar los resultados de generar la dir
adient y de pasarle los argumentos que toquen al generador y ploteador cada saveEvery.
Also los graficos de optimizacion se podrian outputear si el usuario kiere que se outputee todo todo
"""
def SSIM(exp_im,simul):
    mu_simul=simul.mean()
    mu_exp=exp_im.mean()
    var=np.cov(simul.flatten(), exp_im.flatten())
    return ((2*mu_simul*mu_exp+5**2)*(2*var[0,1]+3**2))/((mu_simul**2+mu_exp**2+5**2)*(var[0,0]+var[1,1]+3**2))

class Polarization_Obtention_Algorithm:
    def __init__(self, image_loader, use_exact_gravicenter):
        self.image_names = image_loader.raw_images_names
        self.mode = image_loader.mode
        self.use_exact_gravicenter=use_exact_gravicenter
        self.angles={}
        self.precisions={}

    def reInitialize(self, image_loader):
        self.image_names = image_loader.raw_images_names
        self.angles={}
        self.precisions={}

    def _round_to_sig(self, x_to_round, reference=None, sig=2):
            reference = x_to_round if reference is None else reference
            reference = 1e-13 if reference==0 else reference # to avoid log(0)
            return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)

    def save_images(self, images, output_path, names):
        if type(names) is not list:
            images=[images,]
            names = [names,]
        for name, image in zip(names, images):
            cv2.imwrite(f"{output_path}/{name}.png", image)

    def angle_to_pi_pi(self, angle): # convert any angle to range ()-pi,pi]
        angle= angle%(2*np.pi) # take it to [-2pi, 2pi]
        return angle-np.sign(angle)*2*np.pi if abs(angle)>np.pi else angle

    def set_reference_angle(self, base_reference_angle):
        """
        Use the average of the last computed angle set as a reference.
        base_reference_angle is the angle that the user wants the reference sample to be considered.
        """
        prec=np.mean(list(self.precisions.values()))
        self.reference_precision = self._round_to_sig(prec)
        self.reference_angle = self._round_to_sig(
                            np.mean(list(self.angles.values()))-base_reference_angle,
                                self.reference_precision)


    def process_obtained_angles(self, deg_or_rad=0):
        """
        This is a method to take the obtained angles in the conical refraction image relative to the
        width axis and to process them as:
            - Make them relative to the reference image
            - Undo the doubling of the angle for the projection of the refraction ring
        This method assumes that set_reference_angle() has already been executed before.
        One can choose whether the results should be shown in radian or degrees.
        0 is radian, 1 is deg.
        """
        self.polarization={}
        self.polarization_precision={}
        conv=1 if not deg_or_rad else 180/np.pi # conversion factor
        for name, angle in self.angles.items():
            self.polarization_precision[name] = self._round_to_sig(np.max(self.precisions[name])*conv, self.reference_precision*conv)/2.0
            self.polarization[name]= self._round_to_sig(self.angle_to_pi_pi(angle-self.reference_angle)/2.0*conv, self.polarization_precision[name])

        # TODO: It should be rounded to the significance of the maximum between the reference precision and the precision obtained for the angle



class Radial_Histogram_Algorithm(Polarization_Obtention_Algorithm):
    def __init__(self, image_loader, use_exact_gravicenter, initialize_it=True):
        Polarization_Obtention_Algorithm.__init__(self,image_loader, use_exact_gravicenter)
        self.images = image_loader.centered_ring_images
        if use_exact_gravicenter:
            self.grav = image_loader.g_centered.squeeze() # squeeze for the case ther is only one im
        else: # then use the image center as radial histogram origin
            self.grav = np.array([self.mode]*2)+0.5
        self.min_angle=0
        self.max_angle=2*np.pi
        self.optimals={}
        self.times={}

    def reInitialize(self, image_loader):
        Polarization_Obtention_Algorithm.reInitialize(self, image_loader)
        self.images=image_loader.centered_ring_images
        self.optimals={}
        self.times={}
        if self.use_exact_gravicenter:
            self.grav = image_loader.g_centered.squeeze() # squeeze for the case ther is only one im
        else: # then use the image center as radial histogram origin
            self.grav = np.array([self.mode]*2)+0.5

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
        self.precision = self._round_to_sig(angle_bin_size/2.0)
        self.bin_angles=np.arange(start=self.min_angle, stop=self.max_angle, step=angle_bin_size, dtype=np.float64)[:histograms.shape[1]]+angle_bin_size/2
        for image_name, histogram in zip(self.image_names, histograms):
            # until no stauration images re obtained this is the way (look for the minimum instead of maximum!)
            self.angles[image_name] = self._round_to_sig(self.bin_angles[np.argmin(histogram)]-np.pi, self.precision)
            self.precisions[image_name]=self.precision # redundant!!

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
        self.precision = self._round_to_sig(angle_bin_size/2)
        self.bin_angles=-(angles+self.precision)[:-1] # centers
        for image_name, histogram in zip(self.image_names, histograms):
            self.angles[image_name] = self._round_to_sig(-(angles[np.argmin(histogram)]+angle_bin_size/2)-np.pi, self.precision)
            self.precisions[image_name]=self.precision # redundant!


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
            ax.bar(self.bin_angles, bin_sums, align='center', width=2*self.precision, label=name)
            ax.set_title(f"Optimal angle={self.angles[name]}+-{self.precision} rad\nComputed Bins={bin_sums.shape[0]} . Eff.Time Required={self.times[name]}s")
            ax.set_xlabel("Angle (rad)")
            ax.set_ylabel("Total Intensity")
            #ax.set_yscale('log')
            ax.grid(True)
            plt.savefig(f"{output_path}/Histogram_Algorithm/{title}_{tag}_{name}.png")
            ax.clear()


class Mirror_Flip_Algorithm(Polarization_Obtention_Algorithm):
    def __init__(self, image_loader, min_angle, max_angle, interpolation_flag, initial_guess_delta, method, left_vs_right, use_exact_gravicenter, initialize_it=True):
        """
            Argument image_loader is expected to be an instance of class Image_Loader,
            which has already been initialized and has non-None attributes:
            - self.mode: 203 or 607 depending on whther it contians i607 or i203 images
            - self.centered_ring_images: the [N_images, self.mode*2+1 (h), self.mode*2+1 (w)]
            - self.g_centered: intensity gravicenter in pixel index coordinates [N_images, 2 (h,w)]
            - self.raw_images_names: nems of the N_images in order

        - initial_guess_angle_delta (float): The two initial point ssampled in the middle of the
            initial interval should be almost exactly in the middle, but still slightly separted
            netween them. This will maximize the interval reduction. The present parameter
            will measure their distance.

        """
        Polarization_Obtention_Algorithm.__init__(self,image_loader, use_exact_gravicenter)
        self.images_float = image_loader.centered_ring_images.astype(np.float32)
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

        if use_exact_gravicenter:
            self.grav=image_loader.g_centered #[N_images, 2(h,w)]
        else:
            # gravicenter the same for all
            self.grav=np.array(2*[self.mode])+0.5

    def reInitialize(self, image_loader):
        Polarization_Obtention_Algorithm.reInitialize(self, image_loader)
        self.images_float = image_loader.centered_ring_images.astype(np.float32)
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.times={}
        if self.use_exact_gravicenter:
            self.grav=image_loader.g_centered #[N_images, 2(h,w)]
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
        return udlr*np.sum(np.abs(self.mirror_flip_at_angle(image_array, -angle, center)-image_array)) # we use minus angle to account for mirror flip in w


    def evaluate_mirror_bin(self, image, angle, angles_to_grav, udlr=1):
        # angles in [0, pi/2] are fine
        if angle>0:
            mask=(angles_to_grav>angle) | (angles_to_grav<(angle-np.pi)) #[h,w]
        else:
            mask=(angles_to_grav>angle) & (angles_to_grav<(angle+np.pi)) #[h,w]
        return udlr*np.abs(np.sum(image[mask])-np.sum(image[np.logical_not(mask)]))

    def evaluate_mirror_mask(self, image, angle, cols, rows, center, udlr=1):
        # angles must be in [-pi/2,pi/2] or [3pi/2,2pi]
        mask=np.less(rows, np.tan(-angle)*(cols-center[1])+center[0]) #[h,w] we use minus angle to account for w flip and les instead of greater for the same reason
        return udlr*np.abs(np.sum(image[mask])-np.sum(image[np.logical_not(mask)]))

    def prepare_arguments(self, im):
        if self.method=="mask":
            # avoid this being generated for every image
            self.cols = self.cols if self.cols is not None else np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
            return (self.cols, self.cols.swapaxes(0,1),
                self.grav[im] if self.use_exact_gravicenter else self.grav, self.udlr)

        elif self.method=="bin":
            self.cols = self.cols if self.cols is not None else np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
            # angles relative to the  gravicenter in range [-pi,pi] but for the reversed image
            if self.use_exact_gravicenter:
                self.index_angles = -np.arctan2( self.cols.swapaxes(0,1)-self.grav[im][0], self.cols-self.grav[im][1] ) #[h,w] we set minus because in reality we care for mirror flip in x for coordinate system
            else:
                self.index_angles = self.index_angles if self.index_angles is not None else -np.arctan2( self.cols.swapaxes(0,1)-self.grav[0], self.cols-self.grav[1] ) #[h,w]
            return ( self.index_angles, self.udlr)
        else: # affine
            return (self.grav[im] if self.use_exact_gravicenter else self.grav, self.udlr)

    def given_axis_angle_greater_minus_lower(self, angle, image, center):
        # such that if the output is positive, then R has more intensity and you know immediately that the good angle is the bigger one?
        # de fet esto sugiere un algoritmo con el polano ortogonal que directamente te encuentra el angulo que toca, pero bueno con los que buscan el eje simetrico el truco no parece que funcionara
        self.cols = self.cols if self.cols is not None else np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
        mask=np.less(self.cols.swapaxes(0,1), np.tan(-angle)*(self.cols-center[1])+center[0]) #[h,w] We set -angle, because the coordinates we are thinking of are a mirror flip in w
            # also, we use less instead of greater because we are really thinking on the mirror fliped axes on w
        return np.sum(image[mask])-np.sum(image[np.logical_not(mask)])

    def get_polarization_angle(self, angle, image, center):
        """
        All the mirror methods have the problem that we only get the
        correct angle up to an angle pi. In order to know which is the
        angle to the maximum of the ring (and not the minimum) a final
        subtle check is required.
        """
        #if angle==np.pi or 0: In this case the correct one is not defined by this alg!!!
        if angle==0 or abs(angle)==np.pi:
            angle+=1e-12 # this solution is not ideal, but it works, since we will never get such a good precision
        diff=self.given_axis_angle_greater_minus_lower(angle if self.udlr==-1 else angle+np.pi/2, image, center)
        if self.udlr==-1:
            angle=self.angle_to_pi_pi(angle-np.pi/2)
        if diff>0: # then Upper>Lower -> then good one is the one in (0,pi)
            return angle+np.pi if angle<0 else angle
        else:
            return angle-np.pi if angle>0 else angle


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
        for im, image_name in enumerate(self.image_names):
            name=self.method+f"Brute_Force_{image_name}"
            (self.times[name],
            self.computed_points[name],
            self.optimals[name],
            self.optimums[name],
            self.precisions[name]) = self.optimizer.brute_force_search(
                    angle_steps, zoom_ratios,
                    self.images_float[im], self.prepare_arguments(im))
            self.optimals[name][f"Stage_{len(angle_steps)-1}"] =self.angle_to_pi_pi(self.optimals[name][f"Stage_{len(angle_steps)-1}"])
            self.angles[name]=self.get_polarization_angle(self.optimals[name][f"Stage_{len(angle_steps)-1}"], self.images_float[im],
                self.grav[im] if self.use_exact_gravicenter else self.grav
            )

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

        for im, image_name in enumerate(self.image_names):
            name=f"Fibonacci_Search_{image_name}"
            (self.times[name],
            self.computed_points[name],
            optimal,
            self.optimums[name],
            self.precisions[name])=self.optimizer.fibonacci_ratio_search(
                    precision, maximum_points, cost_tol,
                    self.images_float[im], self.prepare_arguments(im))
            self.optimals[name]=self.angle_to_pi_pi(optimal)
            self.angles[name]=self.get_polarization_angle(self.optimals[name], self.images_float[im],
                self.grav[im] if self.use_exact_gravicenter else self.grav)


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
        for im, image_name in enumerate(self.image_names):
            name=f"Quadratic_Search_{image_name}"
            (self.times[name],
            self.computed_points[name],
            optimal,
            self.optimums[name],
            self.precisions[name])=self.optimizer.quadratic_fit_search(
                precision, max_iterations, cost_tol,
                self.images_float[im], self.prepare_arguments(im))
            self.optimals[name]=self.angle_to_pi_pi(optimal)
            self.angles[name]=self.get_polarization_angle(self.optimals[name], self.images_float[im],
                self.grav[im] if self.use_exact_gravicenter else self.grav)

    def save_result_plots_fibonacci_or_quadratic(self, output_path):
        """
        Save the resulting explored points in cost function vs angle, together with the info
        about the optimization.
        """
        os.makedirs(f"{output_path}/Mirror_Algorithm/", exist_ok=True)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1, 1, 1)
        for name, computed_points in self.computed_points.items():
            ax.plot(computed_points[:,0], computed_points[:,1], 'o', label=name)
            ax.set_title(f"Polarization Angle={self.angles[name]}+-{self.precisions[name]} rad\nOptimal={self.optimals[name]}+-{self.precisions[name]} rad\nComputed Points={len(computed_points)} . Time Required={self.times[name]}s")
            ax.set_xlabel("Angle (rad)")
            ax.set_ylabel("|Mirrorred vs Original|")
            #ax.set_yscale('log')
            ax.grid(True)
            plt.savefig(f"{output_path}/Mirror_Algorithm/{name}.png")
            ax.clear()

    def save_result_plots_brute_force(self, output_path):
        os.makedirs(f"{output_path}/Mirror_Algorithm/", exist_ok=True)
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
                ax.set_ylabel("|Mirrored vs Original|")
                #ax.set_yscale('log')
                ax.grid(True)
            fig.suptitle(f"Polarization angle=Polarization Angle={self.angles[name]}+-{self.precisions[name][f'Stage_{len(axes)-1}']} rad")
            plt.savefig(f"{output_path}/Mirror_Algorithm/{name}.png")
            self.precisions[name]=self.precisions[name][f'Stage_{stage}']


class Gradient_Algorithm(Polarization_Obtention_Algorithm):
    def __init__(self, image_loader, min_radious, max_radious, initial_guess_delta, use_exact_gravicenter, initialize_it=True):
        Polarization_Obtention_Algorithm.__init__(self,image_loader, use_exact_gravicenter)
        self.optimizer = Ad_Hoc_Optimizer(min_radious, max_radious, initial_guess_delta, self.evaluate_mask_radious)
        self.original_images = image_loader
        #self.save_images(self.mirror_images_wrt_width_axis, "./OUTPUT/", [name+"_mirror" for name in self.original_images.raw_images_names])
        self.min_radious = min_radious
        self.max_radious = max_radious
        self.initial_guess_delta = initial_guess_delta
        self.mode = image_loader.mode
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.times={}
        self.masked_gravs={}
        # compute the distance matrices to the gravicenter of the images
        self.cols=np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
        self.rows=self.cols.swapaxes(0,1) #[h,w]
        if initialize_it:
            if use_exact_gravicenter: #[N_images, 2 (h,w)]
                self.grav = image_loader.g_centered.squeeze() # squeeze in case there is only one image
            else: # then use the image center as gravicenter
                self.grav = np.array([self.mode]*2)+0.5


            if not self.use_exact_gravicenter or len(self.grav.shape)==1: #[h,w]
                # all the images have the same mask for distance to center
                self.distances_to_grav = (self.cols-self.grav[0])**2+(self.rows-self.grav[1])**2
            else: # [N_images, h,w]
                self.distances_to_grav = np.array([(self.cols-grav[0])**2+(self.rows-grav[1])**2
                    for grav in self.grav])

    def reInitialize(self, image_loader):
        Polarization_Obtention_Algorithm.reInitialize(self, image_loader)
        self.original_images = image_loader
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.times={}
        self.masked_gravs={}
        if self.use_exact_gravicenter: #[N_images, 2 (h,w)]
            self.grav = image_loader.g_centered.squeeze() # squeeze in case there is only one image
        else: # then use the image center as gravicenter
            self.grav = np.array([self.mode]*2)+0.5

        if not self.use_exact_gravicenter or len(self.grav.shape)==1: #[h,w]
            # all the images have the same mask for distance to center
            self.distances_to_grav = (self.cols-self.grav[0])**2+(self.rows-self.grav[1])**2
        else: # [N_images, h,w]
            self.distances_to_grav = np.array([(self.cols-grav[0])**2+(self.rows-grav[1])**2
                for grav in self.grav])

    def compute_new_gravicenter(self, image, radious, distances_to_grav):
        circle=np.where(distances_to_grav<=radious**2, image, 0) #[h,w]
        # compute the gravicenter of the masked image
        intensity_in_w = np.sum(circle, axis=0) # weights for x [w]
        intensity_in_h = np.sum(circle, axis=1) # weights for y [h]
        total_intensity = intensity_in_h.sum()
        return np.nan_to_num([np.dot(intensity_in_h, np.arange(circle.shape[0]))/total_intensity,
         np.dot(intensity_in_w, np.arange(circle.shape[1]))/total_intensity], nan=self.mode)

    def evaluate_mask_radious(self, image, radious, distances_to_grav, grav):
        # mask the image in the circumference
        circle=np.where(distances_to_grav<=radious**2, image, 0) #[h,w]
        #self.save_images(circle.astype(np.uint8), '.', f"Rad={radious}")
        # compute the gravicenter of the masked image
        intensity_in_w = np.sum(circle, axis=0) # weights for x [w]
        intensity_in_h = np.sum(circle, axis=1) # weights for y [h]
        total_intensity = intensity_in_h.sum()

        new_grav = np.nan_to_num([np.dot(intensity_in_h, np.arange(circle.shape[0]))/total_intensity,
            np.dot(intensity_in_w, np.arange(circle.shape[1]))/total_intensity], nan=self.mode) # The nan to this number works only because the gravicenter is never exctly centered there, else the cost function would yield 0 and the three intial points would be aligned
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
            self.precisions[name]=self.precisions[name][f'Stage_{stage}']

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



class Rotation_Algorithm(Polarization_Obtention_Algorithm):
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
    - Puede pasar el mismo problema que se daba en
    el mirror algorithm, tal que puede dar el ángulo al mínimo en lugar del ángulo al máximo!
    Si quieres considerar ángulos arbitrarios en la búsqueda entonces, deberías hacer la misma
    comprobación final que en el mirror algorithm de hacer el +pi o no.
    """
    def __init__(self, image_loader, min_angle, max_angle, interpolation_flag, initial_guess_delta, use_exact_gravicenter, initialize_it=True):
        """
            Argument image_loader is expected to be an instance of class Image_Loader,
            which has already been initialized and has non-None attributes:
            - self.mode: 203 or 607 depending on whther it contians i607 or i203 images
            - self.centered_ring_images: the [N_images, self.mode*2+1 (h), self.mode*2+1 (w)]
            - self.g_centered: intensity gravicenter in pixel index coordinates [N_images, 2 (h,w)]
            - self.raw_images_names: nems of the N_images in order

        - initial_guess_angle_delta (float): The two initial point ssampled in the middle of the
            initial interval should be almost exactly in the middle, but still slightly separted
            netween them. This will maximize the interval reduction. The present parameter
            will measure their distance.

        """
        Polarization_Obtention_Algorithm.__init__(self,image_loader, use_exact_gravicenter)
        self.original_images = image_loader
        self.interpolation_flag = interpolation_flag
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.initial_guess_delta = initial_guess_delta
        self.mode = image_loader.mode
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.times={}
        self.cols=np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
        self.optimizer = Ad_Hoc_Optimizer(min_angle, max_angle, initial_guess_delta, self.evaluate_image_rotation)
        if initialize_it:
            self.images_float = image_loader.centered_ring_images.astype(np.float32)
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

    def reInitialize(self, image_loader):
        Polarization_Obtention_Algorithm.reInitialize(self, image_loader)
        self.original_images = image_loader
        self.images_float = image_loader.centered_ring_images.astype(np.float32)
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.times={}
        self.cols=np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)) #[h,w]
        if self.use_exact_gravicenter:
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


    def given_axis_angle_greater_minus_lower(self, angle, image, center):
        # such that if the output is positive, then R has more intensity and you know immediately that the good angle is the bigger one?
        # de fet esto sugiere un algoritmo con el polano ortogonal que directamente te encuentra el angulo que toca, pero bueno con los que buscan el eje simetrico el truco no parece que funcionara
        mask=np.less(self.cols.swapaxes(0,1), np.tan(-angle)*(self.cols-center[1])+center[0]) #[h,w] We set -angle, because the coordinates we are thinking of are a mirror flip in w
            # also, we use less instead of greater because we are really thinking on the mirror fliped axes on w
        return np.sum(image[mask])-np.sum(image[np.logical_not(mask)])

    def get_polarization_angle(self, angle, image, center):
        """
        All the mirror methods have the problem that we only get the
        correct angle up to an angle pi. In order to know which is the
        angle to the maximum of the ring (and not the minimum) a final
        subtle check is required.
        """
        #if angle==np.pi or 0: In this case the correct one is not defined by this alg!!!
        if angle==0 or abs(angle)==np.pi:
            angle+=1e-12 # this solution is not ideal, but it works, since we will never get such a good precision
        diff=self.given_axis_angle_greater_minus_lower(angle+np.pi/2, image, center)

        if diff>0: # then Upper>Lower -> then good one is the one in (0,pi)
            return angle+np.pi if angle<0 else angle
        else:
            return angle-np.pi if angle>0 else angle

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
            name=f"Brute_Force_{image_name}"
            (self.times[name],
            self.computed_points[name],
            self.optimals[name],
            self.optimums[name],
            self.precisions[name]) = self.optimizer.brute_force_search(
                    angle_steps, zoom_ratios,
                    self.original_images.centered_ring_images[im], (self.mirror_images_wrt_width_axis[im],
                    self.grav[im] if self.use_exact_gravicenter else self.grav))
            self.optimals[name][f"Stage_{len(angle_steps)-1}"] =self.angle_to_pi_pi(self.optimals[name][f"Stage_{len(angle_steps)-1}"])
            self.angles[name]=self.get_polarization_angle(self.optimals[name][f"Stage_{len(angle_steps)-1}"]/2, self.images_float[im],
                self.grav[im] if self.use_exact_gravicenter else self.grav
            )


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
            name=f"Fibonacci_Search_{image_name}"
            (self.times[name],
            self.computed_points[name],
            optimal,
            self.optimums[name],
            self.precisions[name])=                self.optimizer.fibonacci_ratio_search(
                    precision, maximum_points, cost_tol,
                    self.original_images.centered_ring_images[im], (self.mirror_images_wrt_width_axis[im],
                    self.grav[im] if self.use_exact_gravicenter else self.grav))
            self.optimals[name]=self.angle_to_pi_pi(optimal)
            self.angles[name]=self.get_polarization_angle(self.optimals[name]/2, self.images_float[im],
                self.grav[im] if self.use_exact_gravicenter else self.grav)


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
            name=f"Quadratic_Search_{image_name}"
            (self.times[name],
            self.computed_points[name],
            optimal,
            self.optimums[name],
            self.precisions[name])=self.optimizer.quadratic_fit_search(
                precision, max_iterations, cost_tol,
                self.original_images.centered_ring_images[im], (self.mirror_images_wrt_width_axis[im],
                    self.grav[im] if self.use_exact_gravicenter else self.grav))
            self.optimals[name]=self.angle_to_pi_pi(optimal)
            self.angles[name]=self.get_polarization_angle(self.optimals[name]/2, self.images_float[im],
                self.grav[im] if self.use_exact_gravicenter else self.grav)

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
            ax.set_title(f"Polarization Angle = {self.angles[name]}+-{self.precisions[name]/2} rad\nOptimal={self.optimals[name]}+-{self.precisions[name]} rad\nComputed Points={len(computed_points)} . Time Required={self.times[name]}s")
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
            fig.suptitle(f"Polarization angle=Polarization Angle={self.angles[name]}+-{self.precisions[name][f'Stage_{len(axes)-1}']/2} rad")
            plt.savefig(f"{output_path}/Rotation_Algorithm/{name}.png")
            self.precisions[name]=self.precisions[name][f'Stage_{stage}']


class Simulation_Coordinate_Descent_Algorithm(Polarization_Obtention_Algorithm):
    """
    Execute first the Gradient Algorithm to get an estimation of the angle phi_CR and the radious
    R0 (in pixels) of the CR ring in the image.
    Then using these estimates togther with z=0 and R0=10 execute a coordinate descent in phi_CR,
    R0 pix, R0 mag and Z space to find the simulated CR ring that best fits the experimental image.

    All images can be processed in chunks because we will not fix any stop condition like the
    consecutive difference between found optimals (we could, but as it is so expensive to
    execute the simulations, we will not yet do it -if we are to optimize it all then perhaps
    yes). Instead we will fix a number of coordinate descent cycles for all selected images at
    once!

    USE FLOAT NORMALIZED IMAGES FOR EVERYTHING! ONLY AFTER EVERYTHING DO THE CAST BACK TO UINT!
    """
    def __init__(self, image_loader, coordinate_descent_cycles, min_Z, max_Z, min_phi, max_phi, min_radi, max_radi, min_R0_mag, max_R0_mag, initial_guess_delta_R0, initial_guess_delta_rad, initial_guess_delta_Z, n, w0, a0, max_k, num_k, nx, xChunk, yChunk, gpu, min_radi_G=None, max_radi_G=None, use_exact_gravicenter_G=None, initial_guess_delta_G=None):
        Polarization_Obtention_Algorithm.__init__(self, image_loader, use_exact_gravicenter_G)
        self.cycles=coordinate_descent_cycles
        self.min_Z=min_Z
        self.max_Z=max_Z
        self.min_phi=min_phi
        self.max_phi=max_phi
        self.min_radi=min_radi
        self.max_radi=max_radi
        self.min_R0_mag=min_R0_mag
        self.max_R0_mag=max_R0_mag
        self.min_radi_G=min_radi_G
        self.max_radi_G=max_radi_G
        self.initial_guess_delta_rad=initial_guess_delta_rad
        self.initial_guess_delta_R0=initial_guess_delta_R0
        self.mode=image_loader.mode
        self.interpolation_flag=image_loader.interpolation_flag
        self.image_loader=image_loader
        self.images_normFloat = image_loader.centered_ring_images.astype(np.float64)/np.expand_dims(np.amax(image_loader.centered_ring_images, axis=(1,2)), (1,2))

        if gpu:
            self.simulator=RingSimulator_Optimizer_GPU(n=n, w0=w0, a0=a0, max_k=max_k, num_k=num_k, nx=nx, sim_chunk_x=xChunk, sim_chunk_y=yChunk)
        else:
            self.simulator=RingSimulator_Optimizer(n=n, w0=w0, a0=a0, max_k=max_k, num_k=num_k, nx=nx, sim_chunk_x=xChunk, sim_chunk_y=yChunk)
        if use_exact_gravicenter_G is not None:
            self.algorithm_G = Gradient_Algorithm(image_loader, min_radi_G, max_radi_G, initial_guess_delta_G, use_exact_gravicenter_G)
        else:
            self.algorithm_G=False

        self.angle_optimizer=Ad_Hoc_Optimizer(min_phi, max_phi, initial_guess_delta_rad, self.evaluate_simulation_phi)
        self.radious_optimizer=Ad_Hoc_Optimizer(min_radi, max_radi, initial_guess_delta_R0, self.evaluate_simulation_R0)
        self.R0_mag_optimizer=Ad_Hoc_Optimizer(min_R0_mag, max_R0_mag, initial_guess_delta_Z, self.evaluate_simulation_R0_mag)
        self.Z_optimizer=Ad_Hoc_Optimizer(min_Z, max_Z, initial_guess_delta_Z, self.evaluate_simulation_Z)

        self.times={}
        self.radious_points={}
        self.R0_mag_points={}
        self.Z_points={}
        self.phi_points={}
        # use the results from the gradient algorithm to initialize the best triplet
        self.R0_pix_best={}
        self.R0_mag_best={}
        self.Z_best={}
        self.phi_CR_best={}
        self.last_cycle=coordinate_descent_cycles

        self.best_radii={}
        self.best_R0_mags={}
        self.best_zs={}
        self.best_angles={}
        self.simulations_required={}

    def reInitialize(self, image_loader):
        Polarization_Obtention_Algorithm.reInitialize(self, image_loader)
        self.interpolation_flag=image_loader.interpolation_flag
        self.image_loader=image_loader
        self.images_normFloat = image_loader.centered_ring_images.astype(np.float64)/np.expand_dims(np.amax(image_loader.centered_ring_images, axis=(1,2)), (1,2))

        if self.algorithm_G:
            self.algorithm_G.reInitialize(image_loader)

        self.times={}
        self.radious_points={}
        self.R0_mag_points={}
        self.Z_points={}
        self.phi_points={}
        # use the results from the gradient algorithm to initialize the best triplet
        self.R0_pix_best={}
        self.R0_mag_best={}
        self.Z_best={}
        self.phi_CR_best={}
        self.last_cycle=self.cycles

        self.best_radii={}
        self.best_R0_mags={}
        self.best_zs={}
        self.best_angles={}
        self.simulations_required={}

    def set_reference_angle(self, base_reference_angle):
        """
        Use the average of the last computed angle set as a reference.
        base_reference_angle is the angle that the user wants the reference sample to be considered.
        It is not simple to define a precision here since we are doing coordinate descent
        """
        self.reference_precision = 0
        self.reference_angle = self._round_to_sig(
                            np.mean(list(self.best_angles.values()))-base_reference_angle,
                                self.reference_precision)


    def process_obtained_angles(self, deg_or_rad=0):
        """
        This is a method to take the obtained angles in the conical refraction image relative to the
        width axis and to process them as:
            - Make them relative to the reference image
            - Undo the doubling of the angle for the projection of the refraction ring
        This method assumes that set_reference_angle() has already been executed before.
        One can choose whether the results should be shown in radian or degrees.
        0 is radian, 1 is deg.
        """
        self.polarization={}
        self.polarization_precision={}
        conv=1 if not deg_or_rad else 180/np.pi # conversion factor
        for name, angle in self.best_angles.items():
            self.polarization_precision[name] = 0
            self.polarization[name]= self._round_to_sig(self.angle_to_pi_pi(angle-self.reference_angle)*conv, self.polarization_precision[name])

        # TODO: It should be rounded to the significance of the maximum between the reference precision and the precision obtained for the angle


    def compute_intensity_gravity_center(self, image):
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


    def compute_raw_to_centered_iX(self, image):

        g_raw = self.compute_intensity_gravity_center(image)
        # crop the iamges with size (X+1+X)^2, (607+1+607)^2 or (203+1+203)^2 leaving the gravity center in
        # the central pixel of the image. In case the image is not big enough for the cropping,
        # a 0 padding will be made.
        centered_image = np.zeros(
            (2*self.mode+1, 2*self.mode+1),
            dtype = image.dtype )

        # we round the gravity centers to the nearest pixel indices
        g_index_raw = np.rint(g_raw).astype(int) #[N_images, 2]

        # obtain the slicing indices around the center of gravity
        # TODO -> make all this with a single array operation by stacking the lower and upper in
        # a new axis!!
        # [ 2 (h,w)]
        unclipped_lower = g_index_raw[:]-self.mode
        unclipped_upper = g_index_raw[:]+self.mode+1
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

        # We compute the center of gravity of the cropped images, if everything was made allright
        # they should get just centered in the central pixels number X+1 (idx X), 608 (idx 607) or 204 (idx 203)
        g_centered = self.compute_intensity_gravity_center(centered_image)

        # We now compute a floating translation of the image so that the gravicenter is exactly
        # centered at pixel (607.5, 607.5) (exact center of the image in pixel coordinates staring
        # form (0,0) and having size (607*2+1)x2), instead of being centered at the beginning of
        # around pixel (607,607) as is now
        translate_vectors = self.mode+0.5-g_centered #[ 2(h,w)]
        T = np.float64([[1,0, translate_vectors[1]], [0,1, translate_vectors[0]]])
        return cv2.warpAffine(
                    centered_image, T, (self.mode*2+1, self.mode*2+1),
                    flags=self.interpolation_flag) # interpolation method




    def evaluate_simulation_phi(self, exp_image, angle, R0_pix_best, Z_best, R0_mag_best):
        I=self.compute_raw_to_centered_iX(self.simulator.compute_CR_ring(angle, R0_pix_best, Z_best, R0_mag_best))
        cv2.imwrite(f"ev_phi_phi_{angle}_R0_pix_{R0_pix_best}_Z_{Z_best}_R0_mag_{R0_mag_best}.png", (65535*np.abs(exp_image-I)).astype(np.uint16))
        return np.log(np.sum(np.abs(exp_image-I))+abs(np.sum(exp_image)-np.sum(I))**3)
        #return -SSIM(exp_image, I)

    def evaluate_simulation_R0(self, exp_image, R0_pix, angle_best, Z_best, R0_mag_best):
        I=self.compute_raw_to_centered_iX(
            self.simulator.compute_CR_ring(angle_best, R0_pix, Z_best, R0_mag_best))
        #print(exp_image.max(), I.max(), exp_image.mean(), I.mean())
        cv2.imwrite(f"ev_R0_pix_phi_{angle_best}_R0_pix_{R0_pix}_Z_{Z_best}_R0_mag_{R0_mag_best}.png", (65535*np.abs(exp_image-I)).astype(np.uint16))
        #cv2.imwrite(f"ev_R0_{angle_best}_{R0_pix}_{Z_best}.png", (255*self.compute_raw_to_centered_iX(
        #    self.simulator.compute_CR_ring(angle_best, R0_pix, Z_best))).astype(exp_image.dtype))
        return np.log(np.sum(np.abs(exp_image-I))+abs(np.sum(exp_image)-np.sum(I))**3)
        #return -SSIM(exp_image, I)

    def evaluate_simulation_Z(self, exp_image, Z, angle_best, R0_pix_best, R0_mag_best):
        I=self.compute_raw_to_centered_iX(
            self.simulator.compute_CR_ring(angle_best, R0_pix_best, Z, R0_mag_best))
        return np.log(np.sum(np.abs(exp_image-I))+abs(np.sum(exp_image)-np.sum(I))**3)
        #return -SSIM(exp_image, I)

    def evaluate_simulation_R0_mag(self, exp_image, R0_mag, angle_best, R0_pix_best, Z_best):
        I=self.compute_raw_to_centered_iX(
            self.simulator.compute_CR_ring(angle_best, R0_pix_best, Z_best, R0_mag))
        cv2.imwrite(f"ev_R0_mag_phi_{angle_best}_R0_pix_{R0_pix_best}_Z_{Z_best}_R0_mag_{R0_mag}.png", (65535*np.abs(exp_image-I)).astype(np.uint16))
        return np.log(np.sum(np.abs(exp_image-I))+abs(np.sum(exp_image)-np.sum(I))**3)
        #return -SSIM(exp_image, I)


    def fibonacci_ratio_search(self, precision_pix, maximum_points_R0, precision_phi,
        maximum_points_phi, precision_Z, maximum_points_Z,  precision_R0_mag, maximum_points_R0_mag, cost_tol, precision_G,
        maximum_points_G, cost_tol_G):
        # We first find an estimation for R0 and phiCR using the gradient algorithm
        self.algorithm_G.fibonacci_ratio_search(precision_G, maximum_points_G, cost_tol_G)

        # We now execute a sequence of linear optimizations for each coordinate for every cycle
        for im, name in enumerate(self.image_names):
            self.times[name]={}
            self.radious_points[name]={}
            self.Z_points[name]={}
            self.R0_mag_points[name]={}
            self.phi_points[name]={}
            # use the results from the gradient algorithm to initialize the best triplet
            self.R0_pix_best[name]=[1.65*abs(self.algorithm_G.optimals[f'Fibonacci_Search_{name}'])]
            self.Z_best[name]=[0.0]
            self.R0_mag_best[name]=[10.0]
            self.phi_CR_best[name]=[self.algorithm_G.angles[f'Fibonacci_Search_{name}']]
            simulations=0
            for cycle in range(self.cycles):
                # Optimize Radious magnitude - the width of the croissant
                (time_R0_mag, self.R0_mag_points[name][cycle], R0_mag_best, R0_mag_optimum,
                 _)=self.R0_mag_optimizer.fibonacci_ratio_search(
                    precision_R0_mag, maximum_points_R0_mag, cost_tol,
                    self.images_normFloat[im],
                    (self.phi_CR_best[name][-1], self.R0_pix_best[name][-1], self.Z_best[name][-1]), self.R0_mag_best[name][-1]
                )

                # Optimize Radious
                (time_rad, self.radious_points[name][cycle], R0_pix_best, radi_optimum,
                 _)=self.radious_optimizer.fibonacci_ratio_search(
                    precision_pix, maximum_points_R0, cost_tol,
                    self.images_normFloat[im],
                    (self.phi_CR_best[name][-1], self.Z_best[name][-1], R0_mag_best), self.R0_pix_best[name][-1]
                )

                # Optimize Angle
                (time_ang, self.phi_points[name][cycle], phi_CR_best, phi_optimum, _)=self.angle_optimizer.fibonacci_ratio_search(
                    precision_phi, maximum_points_phi, cost_tol,
                    self.images_normFloat[im],
                    (R0_pix_best, self.Z_best[name][-1], R0_mag_best), self.phi_CR_best[name][-1]
                )

                # Optimize Z
                if maximum_points_Z!=0:
                    self.Z_optimizer.a=-2*np.sqrt(1/3)*R0_mag_best
                    self.Z_optimizer.b=-self.Z_optimizer.a
                    (time_Z, self.Z_points[name][cycle], Z_best, Z_optimum,
                        _)=self.Z_optimizer.fibonacci_ratio_search(
                            precision_Z, maximum_points_Z, cost_tol,
                            self.images_normFloat[im],
                            (phi_CR_best, R0_pix_best, R0_mag_best), self.Z_best[name][-1]
                        )
                else: # dont optimize z
                    time_Z=0
                    Z_best=0
                    Z_optimum=0
                    self.Z_points[name][cycle]=np.array([[0,0,0]])


                self.times[name][cycle]=time_rad+time_ang+time_Z+time_R0_mag
                self.R0_pix_best[name].append(R0_pix_best)
                self.phi_CR_best[name].append(phi_CR_best)
                self.R0_mag_best[name].append(R0_mag_best)
                self.Z_best[name].append(Z_best)

                simulations+=len(self.phi_points[name][cycle])+len(self.radious_points[name][cycle])+len(self.Z_points[name][cycle])+len(self.R0_mag_points[name][cycle])

                # check if convergence criterion is met
                if((abs(Z_optimum-phi_optimum)<cost_tol and
                        abs(phi_optimum-radi_optimum)<cost_tol and
                            abs(radi_optimum-R0_mag_optimum)<cost_tol)
                        or (abs(self.R0_pix_best[name][-2]-R0_pix_best)<precision_pix and
                        abs(self.phi_CR_best[name][-2]-phi_CR_best)<precision_phi and
                        abs(self.Z_best[name][-2]-Z_best)<precision_Z) and
                        abs(self.R0_mag_best[name][-2]-R0_mag_best<precision_R0_mag)):
                    self.last_cycle=cycle+1
                    break
            self.best_radii[name]=R0_pix_best
            self.best_R0_mags[name]=R0_mag_best
            self.best_zs[name]=Z_best
            self.best_angles[name]=self.angle_to_pi_pi(phi_CR_best)
            self.simulations_required[name]=simulations
            logging.info(f"Image {im} optimized!")


    def quadratic_fit_search(self, precision_pix, maximum_points_R0, precision_phi,
        maximum_points_phi, precision_Z, maximum_points_Z, precision_R0_mag, maximum_points_R0_mag, cost_tol, precision_G,
        maximum_points_G, cost_tol_G):
        # We first find an estimation for R0 and phiCR using the gradient algorithm
        self.algorithm_G.quadratic_fit_search(precision_G, maximum_points_G, cost_tol_G)

        # We now execute a sequence of linear optimizations for each coordinate for every cycle
        for im, name in enumerate(self.image_names):
            self.times[name]={}
            self.radious_points[name]={}
            self.Z_points[name]={}
            self.phi_points[name]={}
            self.R0_mag_points[name]={}
            # use the results from the gradient algorithm to initialize the best triplet
            self.R0_pix_best[name]=[1.65*abs(self.algorithm_G.optimals[f'Quadratic_Search_{name}'])]
            self.Z_best[name]=[0.0]
            self.R0_mag_best[name]=[10.0]
            self.phi_CR_best[name]=[self.algorithm_G.angles[f'Quadratic_Search_{name}']]
            simulations=0
            for cycle in range(self.cycles):
                # Optimize Radious magnitude - the width of the croissant
                (time_R0_mag, self.R0_mag_points[name][cycle], R0_mag_best, R0_mag_optimum,
                 _)=self.R0_mag_optimizer.fibonacci_ratio_search(
                    precision_R0_mag, maximum_points_R0_mag, cost_tol,
                    self.images_normFloat[im],
                    (self.phi_CR_best[name][-1], self.R0_pix_best[name][-1], self.Z_best[name][-1]), self.R0_mag_best[name][-1]
                )

                # Optimize Radious
                (time_rad, self.radious_points[name][cycle], R0_pix_best, radi_optimum,
                 _)=self.radious_optimizer.quadratic_fit_search(
                    precision_pix, maximum_points_R0, cost_tol,
                    self.images_normFloat[im],
                    (self.phi_CR_best[name][-1], self.Z_best[name][-1], R0_mag_best), self.R0_pix_best[name][-1]
                )

                # Optimize Angle
                (time_ang, self.phi_points[name][cycle], phi_CR_best, phi_optimum, _)=self.angle_optimizer.quadratic_fit_search(
                    precision_phi, maximum_points_phi, cost_tol,
                    self.images_normFloat[im],
                    (R0_pix_best, self.Z_best[name][-1], R0_mag_best), self.phi_CR_best[name][-1]
                )

                # Optimize Z
                if maximum_points_Z!=0:
                    self.Z_optimizer.a=-2*np.sqrt(1/3)*R0_pix_best*self.simulator.dx
                    self.Z_optimizer.b=-self.Z_optimizer.a
                    (time_Z, self.Z_points[name][cycle], Z_best, Z_optimum,
                        _)=self.Z_optimizer.quadratic_fit_search(
                            precision_Z, maximum_points_Z, cost_tol,
                            self.images_normFloat[im],
                            (phi_CR_best, R0_pix_best, R0_mag_best), self.Z_best[name][-1]
                        )
                else: # dont optimize z
                    time_Z=0
                    Z_best=0
                    Z_optimum=0
                    self.Z_points[name][cycle]=np.array([[0,0,0]])

                self.times[name][cycle]=time_rad+time_ang+time_Z+time_R0_mag
                self.R0_pix_best[name].append(R0_pix_best)
                self.R0_mag_best[name].append(R0_mag_best)
                self.phi_CR_best[name].append(phi_CR_best)
                self.Z_best[name].append(Z_best)

                simulations+=len(self.phi_points[name][cycle])+len(self.radious_points[name][cycle])+len(self.Z_points[name][cycle])+len(self.R0_mag_points[name][cycle])

                # check if convergence criterion is met
                if((abs(Z_optimum-phi_optimum)<cost_tol and
                        abs(phi_optimum-radi_optimum)<cost_tol and
                            abs(radi_optimum-R0_mag_optimum)<cost_tol)
                        or (abs(self.R0_pix_best[name][-2]-R0_pix_best)<precision_pix and
                        abs(self.phi_CR_best[name][-2]-phi_CR_best)<precision_phi and
                        abs(self.Z_best[name][-2]-Z_best)<precision_Z) and
                        abs(self.R0_mag_best[name][-2]-R0_mag_best<precision_R0_mag)):
                    self.last_cycle=cycle+1
                    break
            self.best_radii[name]=R0_pix_best
            self.best_R0_mags[name]=R0_mag_best
            self.best_zs[name]=Z_best
            self.best_angles[name]=self.angle_to_pi_pi(phi_CR_best)
            self.simulations_required[name]=simulations
            logging.info(f"Image {im} optimized!")


    def save_result_plots(self, out_path, meth_name):
        os.makedirs(f"{out_path}/Simulation_Coordinate_Descent_Algorithm/", exist_ok=True)
        self.algorithm_G.save_result_plots_fibonacci_or_quadratic(f"{out_path}/Simulation_Coordinate_Descent_Algorithm/")
        fig = plt.figure(figsize=(4*10, 10*self.last_cycle))

        for k, name in enumerate(self.phi_points.keys()):
            axes = fig.subplots(self.last_cycle, 4)
            if self.last_cycle==1:
                axes=np.expand_dims(axes,0)
            for cycle in range(self.last_cycle):
                axes[cycle, 0].plot(self.R0_mag_points[name][cycle][:,0], self.R0_mag_points[name][cycle][:,1], 'o', label=f'R0 magnitude descent fixing phiCR={self.phi_CR_best[name][cycle]}; R0 pix={self.R0_pix_best[name][cycle]}; Z={self.Z_best[name][cycle]}')
                axes[cycle, 1].plot(self.radious_points[name][cycle][:,0], self.radious_points[name][cycle][:,1], 'o', label=f'R0 pixels descent fixing R0 mag ={self.R0_mag_best[name][cycle+1]}; phiCR={self.phi_CR_best[name][cycle]}; Z={self.Z_best[name][cycle]}')
                axes[cycle, 2].plot(self.phi_points[name][cycle][:,0], self.phi_points[name][cycle][:,1], 'o', label=f'phiCR descent fixing R0 mag ={self.R0_mag_best[name][cycle+1]}; R0 pix={self.R0_pix_best[name][cycle+1]}; Z={self.Z_best[name][cycle]}')
                axes[cycle, 3].plot(self.Z_points[name][cycle][:,0], self.Z_points[name][cycle][:,1], 'o', label=f'Z descent fixing R0 mag ={self.R0_mag_best[name][cycle+1]}; R0 pix={self.R0_pix_best[name][cycle+1]}; phiCR={self.phi_CR_best[name][cycle+1]}')

                axes[cycle, 0].set_xlabel("R0 (magnitude)")
                axes[cycle, 1].set_xlabel("R0 (pixels)")
                axes[cycle, 2].set_xlabel("phi_CR (rad)")
                axes[cycle, 3].set_xlabel("Z (w0-s)")
                axes[cycle, 0].set_title(f"Best of cycle={self.R0_mag_best[name][cycle+1]}\n Computed points={len(self.R0_mag_points[name][cycle])}")
                axes[cycle, 1].set_title(f"Best of cycle={self.R0_pix_best[name][cycle+1]}\n Computed points={len(self.radious_points[name][cycle])}")
                axes[cycle, 2].set_title(f"Best of cycle={self.phi_CR_best[name][cycle+1]}\n Computed points={len(self.phi_points[name][cycle])}")
                axes[cycle, 3].set_title(f"Best of cycle={self.Z_best[name][cycle+1]}\n Computed points={len(self.Z_points[name][cycle])}")

                for i in range(4):
                    axes[cycle, i].set_ylabel("sum(abs(simulated_image(phiCR, R0_pix, R0_mag, Z)-exp_image))")
                    axes[cycle, i].grid(True)
                    axes[cycle, i].legend()

                fig.suptitle(f"{meth_name}  {name}\nBest triplet: phiCR={self.best_angles[name]}rad; R0_pix={self.best_radii[name]}pix; R0_mag={self.best_R0_mags[name]}(a.u); Z={self.best_zs[name]}w0")
            plt.savefig(f"{out_path}/Simulation_Coordinate_Descent_Algorithm/{meth_name}__{name}.png")
            fig.clf()
            I=(65533*self.compute_raw_to_centered_iX(self.simulator.compute_CR_ring(CR_ring_angle=self.best_angles[name], R0_pixels=self.best_radii[name], Z=self.best_zs[name], R0=self.best_R0_mags[name]))).astype(np.uint16)
            cv2.imwrite(f"{out_path}/Simulation_Coordinate_Descent_Algorithm/L_Exp_[{name}]__R_Simul_PolAngle_{self.best_angles[name]/2:.15f}_CRAngle_{self.best_angles[name]:.15f}_Z_{self.best_zs[name]}_R0_pix_{self.best_radii[name]}_R0_mag_{self.best_R0_mags[name]}.png",
                np.concatenate((I, self.image_loader.centered_ring_images[k,:,:]),axis=1))
            #self.simulator.compute_and_plot_CR_ring( CR_ring_angle=self.best_angles[name], R0_pixels=self.best_radii[name], Z=self.best_zs[name], out_path=f"{out_path}/Simulation_Coordinate_Descent/", name=name)
