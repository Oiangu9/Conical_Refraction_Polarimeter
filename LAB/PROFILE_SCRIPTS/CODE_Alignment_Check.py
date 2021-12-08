import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import imageio


main_path="/home/oiangu/Desktop/Conical_Refraction_Polarimeter/DATA/EXPERIMENTAL/Utukuri_et_al/3__29_09_2021/i607"

path_dict={}
for (dirpath, dirnames, filenames) in os.walk(main_path):
    path_dict[dirpath]=filenames


images={}
for dirpath, filenames in path_dict.items():
    for i,filename in enumerate(filenames):
        if 'TRANSITION' not in filename and 'AVERAGE' not in filename and 'PROFILES' not in filename:
            img=cv2.imread(dirpath+'/'+filename, cv2.IMREAD_ANYDEPTH)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                images[filename] = img
                #np.array(Image.open(image_path))
                #Image.open(image_path).show()
            else:
                logging.warning(f" Unable to import image {dirpath+'/'+filename}")
        else:
            filenames.pop(i)

if len(images.values())==0:
    # Then no valid images were introduced!
    logging.error(" No valid images were selected!")


# average image generator
for dirpath, filenames in path_dict.items():
    if len(filenames)>0:
        average_image=np.zeros_like(cv2.imread(dirpath+filenames[0],cv2.IMREAD_ANYDEPTH)).astype(np.float64)
        print(average_image.shape)
        for frame in filenames:
            print(dirpath+frame)
            img = cv2.imread(dirpath+'/'+frame, cv2.IMREAD_ANYDEPTH)
            average_image=average_image+(img.astype(np.float64))/len(filenames)
        cv2.imwrite(f"{dirpath}/AVERAGE_{dirpath.split('/')[-1]}.png", average_image)



# transition gif generator
for dirpath, filenames in path_dict.items():
        if len(filenames)>0:
            with imageio.get_writer(f"{dirpath}/TRANSITION_{dirpath.split('/')[-1]}.gif", mode='I') as writer:
                total_image=imageio.imread(dirpath+'/'+filenames[0])*0.00001
                for frame in filenames:


                    #im=(im+imageio.imread(dirpath+frame)/10).astype(np.uint8)
                    im=imageio.imread(dirpath+'/'+frame)*5
                    #print(np.max(im))
                    #writer.append_data(im)
                    #writer.append_data(im)
                    #writer.append_data(im)
                    writer.append_data(im)
                    writer.append_data(im)
                    writer.append_data(im)



                    """
                    writer.append_data(imageio.imread(dirpath+'/'+frame0))
                    writer.append_data(imageio.imread(dirpath+'/'+frame0))
                    im=(2*(np.abs(imageio.imread(dirpath+'/'+frame0).astype(np.float64)-imageio.imread(dirpath+'/'+frame1).astype(np.float64)))).astype(np.uint8)
                    print(np.max(im))
                    writer.append_data(im)
                    writer.append_data(im)
                    writer.append_data(im)
                    writer.append_data(im)
                    writer.append_data(im)
                    writer.append_data(im)

                    writer.append_data(imageio.imread(dirpath+'/'+frame1))
                    writer.append_data(imageio.imread(dirpath+'/'+frame1))
                writer.append_data(imageio.imread(dirpath+'/'+filenames[-1]))
                """
