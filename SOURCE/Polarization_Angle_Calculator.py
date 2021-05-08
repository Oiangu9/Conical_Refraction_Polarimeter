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
        # centered at pixel (607.5, 607.5) (exact center of hte image in pixel coordinates staring
        # form (0,0) and having size (607*2+1)x2), instead of being centered at the beginning of
        # pixel (607,607) as is now
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
            self.grav = np.array([self.mode*2+1]*2)+0.5
        self.min_angle=0
        self.max_angle=2*np.pi
        self.optimals={}

    def compute_histogram_binning(self, angle_bin_size):
        t=time()
        cols = np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1) ) #[h,w]
        rows = cols.swapaxes(0,1) #[h,w]
        # angles relative to the  gravicenter in range [-pi,pi] but for the reversed image
        index_angles = np.arctan2( rows-self.grav[0], cols-self.grav[1] ) #[h,w]
        # unfortunately in order for the binning to work we will need to have it in 0,2pi (else the 0 bin gets too big)
        print(index_angles.shape)
        index_angles[index_angles<0] += 2*np.pi
        bins = (index_angles//angle_bin_size).astype(int)
        # assign angles to bins and sum intensities of pixels in the same bins
        histograms=np.array([np.bincount( bins.flatten(), im.flatten() ) for im in self.images])
        t=time()-t

        print(histograms.shape, histograms)
        self.histograms=histograms
        self.angles=2*np.pi-np.arange(start=self.min_angle, stop=self.max_angle, step=angle_bin_size, dtype=np.float64)
        for image_name, histogram in zip(self.image_names, histograms):
            self.optimals[image_name] = self.angles[np.argmax(histogram)]+angle_bin_size/2
        self.precisions = angle_bin_size/2
        self.times = t

    def compute_histogram_masking(self, angle_bin_size):
        # if use img center or only one image
        t=time()
        if self.use_exact_gravicenter==False or len(self.grav.shape)==1:
            # create an array with the column number at each element and one with row numbers
            cols = np.expand_dims(np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)),2) #[h,w, 1]
            rows = cols.swapaxes(0,1) #[h,w, 1]
            # note that we set a minus sign for the angles in order to account for the pixel coordinate system and agree with the rest of algorithms (but in reality wrt the image the angles representing them are the same but *-1)
            angles = np.arange(start=-self.max_angle, stop=-self.min_angle, step=angle_bin_size, dtype=np.float64) #[N_theta]
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
            print(histograms)

        else: # then each image has its own masks (they will be slightly different)
            # create an array with the column number at each element and one with row numbers
            cols = np.expand_dims(np.broadcast_to( np.arange(self.mode*2+1), (self.mode*2+1,self.mode*2+1)),(2,3)) #[h,w, 1,1]
            rows = cols.swapaxes(0,1) #[h,w, 1]
            # note that we set a minus sign for the angles in order to account for the pixel coordinate system and agree with the rest of algorithms (but in reality wrt the image the angles representing them are the same but *-1)
            angles = np.arange(start=-self.max_angle, stop=-self.min_angle, step=angle_bin_size, dtype=np.float64) #[N_theta]
            # create masks for each half plane at different orientations
            print("pene1")
            # ESTE ES EL PROBLEMA!!! Bueno, la soluicion es hacer un loop for y hacerlo todo una vez por cada imagen como se hacia antes. Solo pudiendose reutlizar las cosas de hasta aqui.
            greater=np.greater(rows, (np.tan(angles)*((cols-self.grav[:,0]).swapaxes(-2,-1))).swapaxes(-2,-1)+self.grav[:,1]) #[h,w,N_theta, N_images]
            # for angles in [-2pi,-3pi/2] and [-pi/2,0] the mask should be true if col greater than smallest angle and col smaller than greatest angle of bin
            # for angles in [-3pi/2, -pi/2] the mask should be true if smaller than smallest angle and greater than greatest angle of bin
            print("pene2")

            bin_lower = np.concatenate((greater[:,:,angles<-3*np.pi/2,:], np.logical_not(greater)[:,:,(angles>-3*np.pi/2)&(angles<-np.pi/2),:], greater[:,:,angles>-np.pi/2,:]), axis=2) #[h,w,N_theta,N_images]
            #bin_higher = np.logical_not(bin_lower) #[h,w,N_theta,N_images]
            # get the pizza like masks. We have one mask per bin (N_bins=N_theta-1)
            masks=np.logical_and(bin_lower[:,:,:-1,:], np.logical_not(bin_lower)[:,:,1:,:]) # [h,w,N_theta-1,N_images]
            print("pene3")
            histograms=[]
            for im in range(masks.shape[-1]):
                histograms.append([np.sum(self.images[im, masks[:,:,j,im]]) for j in range(angles.shape[0]-1)]) #[N_images, N_theta-1]
            print(histograms)
        t=time()-t
        self.histograms=histograms
        self.angles=-angles
        for image_name, histogram in zip(self.image_names, histograms):
            self.optimals[image_name] = -(angles[np.argmax(histogram)]+angle_bin_size/2)
        self.precisions = angle_bin_size/2
        self.times = t


    def compute_histogram_interpolate(self, angle_bin_size):
        pass

    def refine_by_cosine_fit(self):
        pass

    def save_result_plots(self, output_path):
        "Maybe add the option or check of whether cosine fit should also be plotted"
        pass

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
    def __init__(self, image_loader, min_radious, max_radious, interpolation_flag, initial_guess_delta):
        self.original_images = image_loader
        self.interpolation_flag = interpolation_flag
        self.mirror_images_wrt_width_axis = np.flip(image_loader.centered_ring_images, 1)
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
    def fk_gravicenter_de_mask_centrada_en_g_y_radio_custom(self, image_array, angle):
      pass

    def fk_evaluar_distce_g_c2_en_fk_de_radious_red_circle(self, image_array, angle, mode, reference_image=None):
        pass


    def _round_to_sig(self, x_to_round, reference=None, sig=2):
        reference = x_to_round if reference is None else reference
        return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)
    # y es que el resto es un poco lo mismo, lo que pasa es que aki lo ke optimizarias seria un radio de un numero complejo, que tiene asociado un angulo. Asike aqui el angulo tb esta en cada punto pero vamos, lo ke importara sera el radio en la minimizacion. Tb luego el plot podriamos poner el circulo y tal si acaso, mas que nada para ver que efectivamente el Pogdorf que se obtiene es chachi piruli

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
    def __init__(self, image_loader, min_angle, max_angle, interpolation_flag, initial_guess_delta):
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


    def save_images(self, images, output_path, names):
        if type(names) is not list:
            images=[images,]
            names = [names,]
        for name, image in zip(names, images):
            cv2.imwrite(f"{output_path}/{name}.png", image)


    def rotate_image_by(self, image_array, angle):
      image_center = (self.mode+1.5, self.mode+1.5) # should it be 1.5????
      rot_mat = cv2.getRotationMatrix2D(image_center, angle*180/np.pi, 1.0)
      result = cv2.warpAffine(image_array, rot_mat, image_array.shape, flags=self.interpolation_flag)
      return result.astype(image_array.dtype)


    def evaluate_image_rotation(self, image_array, reference_image,  angle):
        return np.sum(np.abs(self.rotate_image_by(image_array, angle)-reference_image))


    def _round_to_sig(self, x_to_round, reference=None, sig=2):
        reference = x_to_round if reference is None else reference
        return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)


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
        for image_name, image_flip, image_orig in zip(self.original_images.raw_images_names,
                self.mirror_images_wrt_width_axis, self.original_images.centered_ring_images):
            a=self.min_angle
            b=self.max_angle
            # prepare to output data
            self.times[f"Brute_Force_{image_name}"]={}
            self.computed_points[f"Brute_Force_{image_name}"]={}
            self.optimals[f"Brute_Force_{image_name}"]={}
            self.optimums[f"Brute_Force_{image_name}"]={}
            self.precisions[f"Brute_Force_{image_name}"]={}

            # Execute all the stages
            for stage, angle_step in enumerate(angle_steps):
                t=time()
                angles = np.arange(start=a, stop=b, step=angle_step, dtype=np.float64)
                costs = [self.evaluate_image_rotation(image_flip, image_orig, angle) for angle in angles]
                x_min = angles[np.argmin(costs)]
                a=x_min-(b-a)*zoom_ratios[stage]/2.0
                b=x_min+(b-a)*zoom_ratios[stage]/2.0
                t = time()-t

                # output data
                self.times[f"Brute_Force_{image_name}"][f"Stage_{stage}"]=self._round_to_sig(t)
                self.computed_points[f"Brute_Force_{image_name}"][f"Stage_{stage}"]=np.stack((angles, costs)).transpose()
                self.optimals[f"Brute_Force_{image_name}"][f"Stage_{stage}"] = self._round_to_sig(x_min, angle_step)
                self.optimums[f"Brute_Force_{image_name}"][f"Stage_{stage}"] = np.min(costs)
                self.precisions[f"Brute_Force_{image_name}"][f"Stage_{stage}"]=angle_step

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

        # we compute the necessary fibonacci iteration to achieve this precision
        F_Nplus1 = (self.max_angle-self.min_angle)/(2.0*precision)+1
        F_n=[1.0,1.0,2.0]
        # compute fibonacci series till there
        while F_n[-1]<=F_Nplus1:
            F_n.append(F_n[-1]+F_n[-2])

        for image_name, image_flip, image_orig in zip(self.original_images.raw_images_names,
                self.mirror_images_wrt_width_axis, self.original_images.centered_ring_images):
            t=time()
            # prepare the first point triad just like in quadratic fit search
            active_points = self.initialize_correct_point_quad(image_flip, image_orig)
            # for plotting
            computed_points = active_points.transpose().tolist() # list of all the pairs of (xj,f(xj))

            for it in range(len(F_n)-1):
                rho = 1-F_n[-1-it-1]/F_n[-1-it] # interval reduction factor then is (1-rho)
                min_triad = np.argmin(active_points[1]) # 1 or 2 if first 3 or last 3
                if min_triad==1: # we preserve the first 3 points of the quad
                    if rho<1.0/3:
                        active_points[0,-1]=active_points[0,2]-rho*(active_points[0,2]-active_points[0,0])
                    else:
                        active_points[0,-1]=active_points[0,0]+rho*(active_points[0,2]-active_points[0,0])
                    active_points[1,-1] = self.evaluate_image_rotation(image_flip, image_orig, active_points[0,-1])
                    # save new point for plotting
                    computed_points.append(active_points[:,0])
                else: # if 2, we preserve last 3 points
                    if rho>1.0/3:
                        active_points[0,0]=active_points[0,3]-rho*(active_points[0,3]-active_points[0,1])
                    else:
                        active_points[0,0]=active_points[0,1]+rho*(active_points[0,3]-active_points[0,1])
                    active_points[1,0] = self.evaluate_image_rotation(image_flip, image_orig, active_points[0,0])
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
            self.computed_points[f"Fibonacci_{image_name}"] = np.array(computed_points)
            self.optimums[f"Fibonacci_{image_name}"] = active_points[1,min_point]
            self.optimals[f"Fibonacci_{image_name}"] = self._round_to_sig(active_points[0,min_point], ach_precision)
            self.precisions[f"Fibonacci_{image_name}"] = ach_precision
            self.times[f"Fibonacci_{image_name}"]=self._round_to_sig(t)



    def initialize_correct_point_quad(self, image_flip, image_orig):
        """
        We initialize a point quad where the minimum of the cost function is for sure not in the
        boundaries of the quad.
        This is necessary for the quadratic fit search, and at least convenient for the fibonacci
        search.

        Returns an array [2,4] with the zero-th row having the ordered angles and the first row
        their cost function values, such that the minimum cost of the four pairs of points
        is in position 1 or 2.
        """
        # Initialize the active points to consider in the first iteration
        active_xs = np.array([self.min_angle,
                                0.5*(self.max_angle+self.min_angle-self.initial_guess_delta),
                                0.5*(self.max_angle+self.min_angle+self.initial_guess_delta),
                                self.max_angle], dtype=np.float64)

        # Evaluate cost function for each angle
        active_points = np.stack((active_xs, [ self.evaluate_image_rotation(image_flip, image_orig, angle) for angle in active_xs])) # [2 (xj,f(xj)),4]

        # if the minium is in the boundary of the interval, make it not be the boundary
        if np.argmin(active_points[1])==0:
            active_points[0, 3] -= 3*np.pi
            active_points[1,3] = self.evaluate_image_rotation(image_flip, image_orig, active_points[0, 3])
        elif np.argmin(active_points[1])==3:
            active_points[0, 0] += 3*np.pi
            active_points[1,0] = self.evaluate_image_rotation(image_flip, image_orig, active_points[0,0])
        # order the four pairs of points by their angle
        return active_points[:, np.argsort(active_points[0])]

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
        for image_name, image_flip, image_orig in zip(self.original_images.raw_images_names,
                self.mirror_images_wrt_width_axis, self.original_images.centered_ring_images):
            t=time()
            it=0

            active_points = self.initialize_correct_point_quad( image_flip, image_orig)

            computed_points = active_points.transpose().tolist() # list of all the pairs of (xj,f(xj))

            while( np.abs(active_points[0,-1]-active_points[0,0]) >= 2*precision and not np.allclose(active_points[1,:], active_points[1,0], rtol=cost_tol) and it<=max_iterations):
                # Choose new triad of angles
                min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
                # using the fact that the minimum of the four points will never be in the boundary
                active_points[:, :3] = active_points[:, (min_point-1):(min_point+2)]

                # compute the interpolation polynomial parameters and the minimum
                x_min = 0.5*( active_points[0,0]+active_points[0,1] + (active_points[0,0]-active_points[0,2])*(active_points[0,1]-active_points[0,2])/( ( active_points[0,0]*(active_points[1,2]-active_points[1,1])+active_points[0,1]*(active_points[1,0]-active_points[1,2]) )/(active_points[1,1]-active_points[1,0]) + active_points[0,2] ) )
                active_points[0,3] = x_min
                active_points[1,3] = self.evaluate_image_rotation(image_flip, image_orig, x_min)

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
            self.computed_points[f"Quadratic_{image_name}"] = np.array(computed_points)
            self.optimums[f"Quadratic_{image_name}"] = active_points[1,min_point]
            self.optimals[f"Quadratic_{image_name}"] = self._round_to_sig(active_points[0,min_point], ach_precision)
            self.precisions[f"Quadratic_{image_name}"] = ach_precision
            self.times[f"Quadratic_{image_name}"]=self._round_to_sig(t)



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
