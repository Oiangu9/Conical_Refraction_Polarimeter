import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from skimage.measure import block_reduce # useful function for average pooling
import os
from time import time
#from PIL import Image


"""
    TOPONDER
    -------
    Claro, que es mejor, tener los paths de las imagenes e ir procesnadolas de uno en uno, pues
    claro, si te inputea 1000 imagenes pues la cagas, oo tenerlas todas en un dictionary ya
    importadas y trabajar en ellas de una en una, oo tenerlas todas en un array numpy de forma
    que puedas todo el array a la vez (a todas las imagenes a la vez) aplicarles todos los cambios.

    Lo más eficiente parece ser lo último. Para eso though, será importante mantener una lista
    con los nombres de los ficheros ordenados exactamente como están las imágenes en el tensor.

    Tmabién si es así, es muy importante que todas las imágenes de input deben tener la misma
    dimensionalidad (aunque no tengan porqué tener el mismo formato claro.)
    Se pueden hacer estas comprobaciones si todor lo pide, y hacerlo todo por separado hasta tener
    las imagenes i607, momento en el que ya sí que puedes juntarlos todos en un solo tensor.


    TODO
    ----
    > Mira los apuntes d eoptimizacion y planifica claramente cada uno de los tres minimization
        algorithms
    > Implementalos y haz los paramteros definition y tal
    > Piensa como se puede hace run histograma angular desde el centro, cómo hacer máscaras de
        indexing que tengan forma así, y piensa lo de las áreas para ponderarlos.
    > En ese caso e ssimple porque si los computas los valores del histograma, ya tienes cual es el
        bin mínimo con su incertidumbre ames. Y puedes poner de parametero o el numero de bins
        or even better el ancho en angulo que cada bin va a contener!

"""

class Image_Loader:
    def __init__(self, mode, interpolation_flag):
        """
            Mode is expected to be 607 or 203, depending of whether i607 or i203 is desired to be
            used for all the algorithms.
        """
        self.mode = mode
        self.interpolation_flag = interpolation_flag

    def get_raw_images_to_compute(self, path_list):
        """
            A list of full paths (strings) is expected with the images in it.
            This will import all the raw images into numpy arrays, ready to be converted into
            i607 or i203. In fact, in the first verision, we will not check if they have the
            same dimensions, and we will simply save them all into a single numpuy tensor for
            the sake of efficiency.

            self.raw_images : [N_images, raw_height, raw_width]

        """
        logging.info("\n> Importing Images one by one...\n")
        images={}
        # We take the images that are valid from the provided paths and convert them to grayscale
        for image_path in path_list:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[image_path] = img
                #np.array(Image.open(image_path))
                #Image.open(image_path).show()
            else:
                logging.warning(f" Unable to import image {image_path}")
        if len(images.keys())==0:
            # Then no valid images were introduced!
            logging.error(" No valid images were selected!")
            return 1
        # We assume they have the same width and height as (height, width) tuple
        self.raw_image_shape = next(iter(images.values())).shape

        # We allocate a numpy array to save all the images
        self.raw_images = np.zeros((len(images),)+self.raw_image_shape,
            dtype=next(iter(images.values())).dtype) # important to maintain data types!
        self.raw_images_names = []
        # Now generate the ordered list with the names and the image tensor
        for k, (image_path, image_array) in enumerate(images.items()):
            self.raw_images_names.append(image_path.rsplit('/',1)[-1].rsplit('.',1)[0])
            self.raw_images[k,:,:] = image_array

        if(self.mode==203): # the mode is set to 203, we will need to downscale the raw image by 3,
                            # in order for the i203 to be able to contain the whole ring
            self.raw_images = block_reduce(self.raw_images,
                block_size=(1,3, 3), func=np.mean).astype(self.raw_images.dtype)
            self.raw_image_shape = self.raw_images.shape[1:]

        logging.info(f"\n Imported {len(self.raw_images_names)} images of size "+
            f"{self.raw_image_shape} into a {self.raw_images.shape} numpy tensor {self.raw_images.dtype}")

        '''
        # Fast show of the imported images. For this we will need to rescale them. this code can
        # be deleted when everything is OK
        resize_factor=4
        h,w = self.raw_image_shape
        h = int(h / resize_factor)  #  one must compute beforehand
        w = int(w / resize_factor)  #  and convert to INT
        for i in range(len(self.raw_images_names)):
            #plt.imshow(self.raw_images[i,:,:])
            #plt.show()
            cv2.imshow(f'Inputed Image {self.raw_images_names[i]}',
                cv2.resize(self.raw_images[i,:1709,:2336], (w,h)))
            ok = cv2.waitKey(500)
            cv2.destroyAllWindows()
        '''


    def compute_intensity_gravity_center(self, images):
        """
            Expects input images to be an array of dimensions [N_images, h, w].
            It will return an array of gravity centers [N_images, 2(h,w)] in pixel coordinates
            Remember that pixel coordinates are set equal to numpy indices, so they being at 0

        """
        # image wise total intensity and marginalized inensities for weighted sum
        # (preserves axis 0, where images are stacked)
        intensity_in_w = np.sum(images, axis=1) # weights for x [N_imgs, raw_width]
        intensity_in_h = np.sum(images, axis=2) # weights for y [N_imgs, raw_height]
        total_intensity = intensity_in_h.sum(axis=1)

        # Compute mass center for intensity (in each image axis)
        # [N_images, 2] (h_center,w_center)
        return np.stack(
            (np.dot(intensity_in_h, np.arange(images.shape[-2]))/total_intensity,
             np.dot(intensity_in_w, np.arange(images.shape[-1]))/total_intensity)
            ).transpose()


    def compute_raw_to_i607_or_i203(self, output_path):
        """
            Computes the converison to i607 or i203 and saves the resulting images as
            png in the output directory provided as argument.

            The function assumes the desired images to be converted are saved in self.raw_images.
            This numpy array will be freed once thisfunction is computed in order to save RAM.

        """
        g_raw = self.compute_intensity_gravity_center(self.raw_images)

        logging.info(f" \nCenters of Intensity gravity in raw pixel coordinates: {g_raw}")
        # Cropear after padding y computa again el centro de masas -ein funkiño bat izetie-
        # Aplica una translacion en coordenadas proyectovas por t1,t2 pa centrarlo exactamente

        # crop the iamges with size (607+1+607)^2 or (203+1+203)^2 leaving the gravity center in
        # the central pixel of the image. In case the image is not big enough for the cropping,
        # a 0 padding will be made.
        self.centered_ring_images = np.zeros(
            (g_raw.shape[0], 2*self.mode+1, 2*self.mode+1),
            dtype = self.raw_images.dtype )

        # we round the gravity centers to the nearest pixel indices
        g_index_raw = np.rint(g_raw).astype(int) #[N_images, 2]

        # obtain the slicing indices around the center of gravity
        # TODO -> make all this with a single array operation by stacking the lower and upper in
        # a new axis!!
        # [N_images, 2 (h,w)]
        unclipped_lower = g_index_raw[:]-self.mode
        unclipped_upper = g_index_raw[:]+self.mode+1
        # unclippde could get out of bounds for the indices, so we clip them
        lower_bound = np.clip( unclipped_lower, a_min=0, a_max=self.raw_image_shape)
        upper_bound = np.clip( unclipped_upper, a_min=0, a_max=self.raw_image_shape)
        # we use the difference between the clipped and unclipped to get the necessary padding
        # such that the center of gravity is left still in the center of the images
        padding_lower = lower_bound-unclipped_lower
        padding_upper = upper_bound-unclipped_upper

        # crop the images
        for im in range(g_raw.shape[0]):
            self.centered_ring_images[im, padding_lower[im,0]:padding_upper[im,0] or None,
                                        padding_lower[im,1]:padding_upper[im,1] or None ] = \
                      self.raw_images[im, lower_bound[im,0]:upper_bound[im,0],
                                          lower_bound[im,1]:upper_bound[im,1]]

        # We compute the center of gravity of the cropped images, if everything was made allright
        # they should get just centered in the central pixels number 608 (idx 607) or 204 (idx 203)
        g_centered = self.compute_intensity_gravity_center(self.centered_ring_images)
        #logging.info(f"\n Intensity gravicenter in i{self.mode} images: {g_centered}, sizes {self.centered_ring_images.shape}")

        # We now compute a floating translation of the image so that the gravicenter is exactly
        # centered at pixel (607.5, 607.5) (exact center of the image in pixel coordinates staring
        # form (0,0) and having size (607*2+1)x2), instead of being centered at the beginning of
        # around pixel (607,607) as is now
        translate_vectors = self.mode+0.5-g_centered #[N_images, 2(h,w)]
        for im in range(g_raw.shape[0]):
            T = np.float64([[1,0, translate_vectors[im, 1]], [0,1, translate_vectors[im, 0]]])
            self.centered_ring_images[im] = cv2.warpAffine(
                        self.centered_ring_images[im], T, (self.mode*2+1, self.mode*2+1),
                        flags=self.interpolation_flag) # interpolation method
            cv2.imwrite(f"{output_path}/{self.raw_images_names[im]}.png", self.centered_ring_images[im])

        # We recompute the gravity centers:
        self.g_centered = self.compute_intensity_gravity_center(self.centered_ring_images)

        logging.info(f"\n Fine-tuned intensity gravicenter in i{self.mode} images: {self.mode+0.5-self.g_centered}, sizes {self.centered_ring_images.shape}")

        # Remove the raw images
        del self.raw_images
        del self.raw_image_shape


    def import_converted_images(self, path_list):
        """
        Instead of computing the i607or i203 images form raw images, we could also import
        them directly, already converted. path_list should provide a list with the paths
        to converted images of the self.mode kind.

        """

        images={}
        for image_path in path_list:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[image_path] = img
                #np.array(Image.open(image_path))
                #Image.open(image_path).show()
            else:
                logging.warning(f" Unable to import image {image_path}")
        if len(images.values())==0:
            # Then no valid images were introduced!
            logging.error(" No valid images were selected!")
            return 1
        self.centered_ring_images = np.zeros(
                (len(images.keys()), 2*self.mode+1, 2*self.mode+1),
                dtype = next(iter(images.values())).dtype )
        self.raw_images_names=[]
        for k, (image_path, image_array) in enumerate(images.items()):
            if image_array.shape[0]==self.mode*2+1:
                self.raw_images_names.append(image_path.rsplit('/',1)[-1].rsplit('.',1)[0])
                self.centered_ring_images[k,:,:] = image_array

        if len(self.raw_images_names)==0:
            # Then no valid images were introduced!
            logging.error(" No valid images were selected!")
            return 1
        # We recompute the gravity centers [N_images, 2 (h,w)]
        self.g_centered = self.compute_intensity_gravity_center(self.centered_ring_images)



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
    def __init__(self, image_loader, min_angle, max_angle, interpolation_flag, initial_guess_delta):
        self.original_images = image_loader
        self.interpolation_flag = interpolation_flag
        self.mirror_images_wrt_width_axis = np.flip(image_loader.centered_ring_images, 1)
        #self.save_images(self.mirror_images_wrt_width_axis, "./OUTPUT/", [name+"_mirror" for name in self.original_images.raw_images_names])
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.initial_guess_delta = initial_guess_delta
        self.mode = image_loader.mode
        self.computed_points={}
        self.optimums={}
        self.optimals={}
        self.precisions={}
        self.times={}
    def mirror_flip_at_angle(self, image_array, angle):
      pass

    def evaluate_image_flip(self, image_array, angle, mode, reference_image=None):
        pass


    def _round_to_sig(self, x_to_round, reference=None, sig=2):
        reference = x_to_round if reference is None else reference
        return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)

    # Y el resto de fknes habra que copy pastearlas e ir cambiando y viendo si merece + la pena kisas ponerlas en una misma clase

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



class Ad_Hoc_Optimizer:
    def __init__(self, min_in_range, max_in_range, initial_guess_delta, evaluate_cost, fib_prec=None):
        self.a=min_in_range
        self.b=max_in_range
        self.initial_guess_delta = initial_guess_delta
        self.evaluate_cost=evaluate_cost
        if fib_prec is None:
            self.F_n=None
        else:
            self.compute_fib_iteration_for_prec(fib_prec)

    def _round_to_sig(self, x_to_round, reference=None, sig=2):
        reference = x_to_round if reference is None else reference
        return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)

    def brute_force_search(self, steps, zoom_ratios, image, args_for_cost):
        """
        Arguments
        --------
        - steps (list): A list of the different feasible point steps to take in each of the sweeps.
            Expected N, where N is the number of sweeps that will be performed. The first one is
            expected to be the coarsest grain and they should be ordered from big to smallest.
            The last step in the list will define the precision of the found minimum. The point
            steps are expected to be in (0,b-a).

        - zoom_ratios (list): A list of the interval reductions that will be held after each sweep
            around the current best candidate for the minimum. There should be N-1 elements and
            they should be numbers in (0,1]. It should have an extra last element to account for
            the last iteration.
        Returns
        -------
        time, computed_points, optimal, optimum, precision
        """
        if not isinstance(args_for_cost, (list, tuple)):
            args_for_cost = (args_for_cost,)
        optimals={}
        optimums={}
        times={}
        precisions={}
        computed_points={}
        a, b=self.a, self.b
        # Execute all the stages
        for stage, step in enumerate(steps):
            t=time()
            feasible_points = np.arange(start=a, stop=b, step=step, dtype=np.float64)
            costs = [self.evaluate_cost(image, point, *args_for_cost) for point in feasible_points]
            x_min = feasible_points[np.argmin(costs)] # pero seria maximo
            a=x_min-(b-a)*zoom_ratios[stage]/2.0
            b=x_min+(b-a)*zoom_ratios[stage]/2.0
            t = time()-t
            times[f"Stage_{stage}"]=self._round_to_sig(t)
            computed_points[f"Stage_{stage}"]=np.stack((feasible_points, costs)).transpose()
            optimals[f"Stage_{stage}"] = self._round_to_sig(x_min, step)
            optimums[f"Stage_{stage}"] = np.min(costs)
            precisions[f"Stage_{stage}"]=step
        return times, computed_points, optimals, optimums, precisions

    def compute_fib_iteration_for_prec(self, precision):
        # we compute the necessary fibonacci iteration to achieve this precision
        F_Nplus1 = (self.b-self.a)/(2.0*precision)+1
        self.F_n=[1.0,1.0,2.0]
        # compute fibonacci series till there
        while self.F_n[-1]<=F_Nplus1:
            self.F_n.append(self.F_n[-1]+self.F_n[-2])

    def fibonacci_ratio_search(self, precision, maximum_points, cost_tol, image, args_for_cost):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            cost evaluation arithmetics.
            Noise or discretization can induce plateaus in the minimum.

            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision
            will be outputed accordingly.

        - maximum_points (int): Maximum number of points to use in the minimum search. It is also
            the number of times to make an interval reduction.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        Returns
        -------
        time, computed_points, optimal, optimum, precision
        """
        if not isinstance(args_for_cost, (list, tuple)):
            args_for_cost = (args_for_cost,)
        if self.F_n is None:
            self.compute_fib_iteration_for_prec(precision)

        t=time()
        # prepare the first point triad just like in quadratic fit search
        active_points = self.initialize_correct_point_quad(image, args_for_cost)
        # for plotting
        computed_points = active_points.transpose().tolist() # list of all the pairs of (xj,f(xj))

        for it in range(len(self.F_n)-1):
            rho = 1-self.F_n[-1-it-1]/self.F_n[-1-it] # interval reduction factor then is (1-rho)
            min_triad = np.argmin(active_points[1]) # 1 or 2 if first 3 or last 3
            if min_triad==1: # we preserve the first 3 points of the quad
                if rho<1.0/3:
                    active_points[0,-1]=active_points[0,2]-rho*(active_points[0,2]-active_points[0,0])
                else:
                    active_points[0,-1]=active_points[0,0]+rho*(active_points[0,2]-active_points[0,0])
                active_points[1,-1] = self.evaluate_cost(image, active_points[0,-1], *args_for_cost)
                # save new point for plotting
                computed_points.append(active_points[:,0])
            else: # if 2, we preserve last 3 points
                if rho>1.0/3:
                    active_points[0,0]=active_points[0,3]-rho*(active_points[0,3]-active_points[0,1])
                else:
                    active_points[0,0]=active_points[0,1]+rho*(active_points[0,3]-active_points[0,1])
                active_points[1,0] = self.evaluate_cost(image, active_points[0,0], *args_for_cost)
                # save new point for plotting
                computed_points.append(active_points[:,0])

            # order the four pairs of points by their angle
            active_points = active_points[:, np.argsort(active_points[0])]

            if np.abs(active_points[0,-1]-active_points[0,0]) < 2*precision or np.allclose(active_points[1,:], active_points[1,0], rtol=cost_tol) or it==maximum_points:
                break
        t = time()-t
        # save all the data
        min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
        ach_precision = self._round_to_sig((active_points[0, min_point+1]-active_points[0, min_point-1])/2.0)

        return (self._round_to_sig(t), np.array(computed_points),
            self._round_to_sig(active_points[0,min_point], ach_precision),
            active_points[1,min_point], ach_precision)

    def initialize_correct_point_quad(self, image, args_for_cost):
        """
        We initialize a point quad where the minimum of the cost function is for sure not in the
        boundaries of the quad.
        This is necessary for the quadratic fit search, and at least convenient for the fibonacci
        search.

        Returns an array [2,4] with the zero-th row having the ordered feasible points
        and the first row their cost function values, such that the minimum cost of the four
        pairs of points is in position 1 or 2.
        """
        # Initialize the active points to consider in the first iteration
        active_xs = np.array([self.a,
                                0.5*(self.b+self.a-self.initial_guess_delta),
                                0.5*(self.b+self.a+self.initial_guess_delta),
                                self.b], dtype=np.float64)

        # Evaluate cost function for each angle
        active_points = np.stack((active_xs, [ self.evaluate_cost(image, angle, *args_for_cost) for angle in active_xs])) # [2 (xj,f(xj)),4]
        # if the minium is in the boundary of the interval, make it not be the boundary
        if np.argmin(active_points[1])==0:
            active_points[0, 3] -= 3*(self.b-self.a)/2
            active_points[1,3] = self.evaluate_cost(image, active_points[0, 3], *args_for_cost)
        elif np.argmin(active_points[1])==3:
            active_points[0, 0] += 3*(self.b-self.a)/2
            active_points[1,0] = self.evaluate_cost(image, active_points[0,0], *args_for_cost)

        # order the four pairs of points by their support position
        return active_points[:, np.argsort(active_points[0])]

    def quadratic_fit_search(self, precision, max_iterations, cost_tol, image, args_for_cost):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the function. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            cost eval. arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision
            will be outputed accordingly.

        - max_iterations (int): Number of maximum iterations of quadratic function fit and
            minimization to tolerate.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        Returns
        -------
        time, computed_points, optimal, optimum, precision

        """
        if not isinstance(args_for_cost, (list, tuple)):
            args_for_cost = (args_for_cost,)
        t=time()
        it=0
        active_points = self.initialize_correct_point_quad( image, args_for_cost)

        computed_points = active_points.transpose().tolist() # list of all the pairs of (xj,f(xj))

        while( np.abs(active_points[0,-1]-active_points[0,0]) >= 2*precision and not np.allclose(active_points[1,:-1], active_points[1,1], rtol=cost_tol) and it<=max_iterations):
            # Choose new triad of angles
            min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
            # using the fact that the minimum of the four points will never be in the boundary
            active_points[:, :3] = active_points[:, (min_point-1):(min_point+2)]
            # compute the interpolation polynomial parameters and the minimum
            x_min = 0.5*( active_points[0,0]+active_points[0,1] + (active_points[0,0]-active_points[0,2])*(active_points[0,1]-active_points[0,2])/( ( active_points[0,0]*(active_points[1,2]-active_points[1,1])+active_points[0,1]*(active_points[1,0]-active_points[1,2]) )/(active_points[1,1]-active_points[1,0]) + active_points[0,2] ) )
            active_points[0,3] = x_min
            active_points[1,3] = self.evaluate_cost(image, x_min, *args_for_cost)

            # save new point for plotting
            computed_points.append(active_points[:,3])

            # order the four pairs of points by their angle
            active_points = active_points[:, np.argsort(active_points[0])]
            # increment iterations
            it+=1

        t = time()-t
        # save al the data
        min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
        ach_precision = self._round_to_sig((active_points[0, min_point+1]-active_points[0, min_point-1])/2.0)
        return (self._round_to_sig(t), np.array(computed_points),
            self._round_to_sig(active_points[0,min_point], ach_precision),
            active_points[1,min_point], ach_precision)
