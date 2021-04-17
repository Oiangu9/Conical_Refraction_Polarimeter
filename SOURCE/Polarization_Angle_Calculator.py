import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from skimage.measure import block_reduce # useful function for average pooling

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
    def __init__(self, mode):
        """
            Mode is expected to be 607 or 203, depending of whether i607 or i203 is desired to be
            used for all the algorithms.
        """
        self.mode = mode

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

        # We assume they have the same width and height as (height, width) tuple
        self.raw_image_shape = next(iter(images.values())).shape

        # We allocate a numpy array to save all the images
        self.raw_images = np.zeros((len(images),)+self.raw_image_shape,
            dtype=next(iter(images.values())).dtype) # important to maintain data types!
        self.raw_images_names = []
        # Now generate the ordered list with the names and the image tensor
        for k, (image_path, image_array) in enumerate(images.items()):
            self.raw_images_names.append(image_path.rsplit('/',1)[-1])
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
                            self.centered_ring_images[im], T, (self.mode*2+1, self.mode*2+1))
            cv2.imwrite(f"{output_path}/{self.raw_images_names[im]}", self.centered_ring_images[im])

        # We recompute the gravity centers:
        self.g_centered = self.compute_intensity_gravity_center(self.centered_ring_images)

        logging.info(f"\n Fine-tuned intensity gravicenter in i{self.mode} images: {self.mode+0.5-self.g_centered}, sizes {self.centered_ring_images.shape}")

        # Remove the raw images
        del self.raw_images
        del self.raw_image_shape



class Rotation_Algorithm:
    """
    The distance between the images, i607 and rotated i607R as a function of the rotation angle,
    once both images are centered in the gravicenter is a strictly unimodal and convex function,
    the minimum of which should be extremely simple to find (disregarding the fluctuations that
    might appear in the microscopic rotation scale due to the discrete nature of images or the
    lossy compression issues). A simple algorithm that can proove to be good enough is the Golden
    Ratio three point search.

    Brute force grid search:
    -   Hace run grid de step tal grosso, buscar los 3 ptos minimos, coger el bounding box y hacer
        un grid mas fino con tantos puntos como antes. Y otra vez. Aka haciendo zoom ins. Asi hasta
        que la distancia en metrica entre los tres puntos sea menor a una tolerancia epsilon. En
        cada iteracion podemos plotear el grafico.

    - Golden ratio approach. Tope simple y rapido. Guardo los puntos que salgan para un plot final.

    - Y yet another que es even better existe casi seguro. Algo con una componente mas estocastica.

    """
    def __init__(self, image_loader):
        """
            Argument image_loader is expected to be an instance of class Image_Loader,
            which has already been initialized and has non-null attributes:
            - self.mode: 203 or 607 depending on whther it contians i607 or i203 images
            - self.centered_ring_images: the [N_images, self.mode*2+1, self.mode*2+1]
            - self.g_centered: intensity gravicenter in pixel index coordinates [N_images, 2 (h,w)]

        """
        self.image_loader = image_loader
