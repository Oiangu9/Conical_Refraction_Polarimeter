import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
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

"""

class Polarization_Angle_Calculator:
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
        self.raw_images_paths = []
        # Now generate the ordered list with the names and the image tensor
        for k, (image_path, image_array) in enumerate(images.items()):
            self.raw_images_paths.append(image_path)
            self.raw_images[k,:,:] = image_array

        logging.info(f"\n Imported {len(self.raw_images_paths)} images of size "+
            f"{self.raw_image_shape} into a {self.raw_images.shape} numpy tensor {self.raw_images.dtype}")

        '''
        # Fast show of the imported images. For this we will need to rescale them. this code can
        # be deleted when everything is OK
        resize_factor=4
        h,w = self.raw_image_shape
        h = int(h / resize_factor)  #  one must compute beforehand
        w = int(w / resize_factor)  #  and convert to INT
        for i in range(len(self.raw_images_paths)):
            #plt.imshow(self.raw_images[i,:,:])
            #plt.show()
            cv2.imshow(f'Inputed Image {self.raw_images_paths[i]}',
                cv2.resize(self.raw_images[i,:1709,:2336], (w,h)))
            ok = cv2.waitKey(500)
            cv2.destroyAllWindows()
        '''

    def compute_raw_to_i607_or_i203(self, output_path):
        """
            Computes the converison to i607 or i203 and saves the resulting images as
            png in the output directory provided as argument.

            The function assumes the desired images to be converted are saved in self.raw_images.
            This numpy array will be freed once thisfunction is computed in order to save RAM.

        """
        # image wise total intensity and marginalized inensities for weighted sum
        # (preserves axis 0, where images are stacked)
        intensity_in_w = np.sum(self.raw_images, axis=1) # weights for x [N_imgs, raw_width]
        intensity_in_h = np.sum(self.raw_images, axis=2) # weights for y [N_imgs, raw_height]
        total_intensity = intensity_in_h.sum(axis=1)


        # Compute mass center for intensity (in each image axis) [N_images, 2] (h_center,w_center)
        g_raw = np.stack(
            (np.dot(intensity_in_h, np.arange(self.raw_image_shape[0]))/total_intensity,
             np.dot(intensity_in_w, np.arange(self.raw_image_shape[1]))/total_intensity)
             ).transpose()

        logging.info(f" Centers of Intensity gravity in raw pixel coordinates: {g_raw}")
        # Cropear after padding y computa again el centro de masas -ein funkiño bat izetie-
        # Aplica una translacion en coordenadas proyectovas por t1,t2 pa centrarlo exactamente
