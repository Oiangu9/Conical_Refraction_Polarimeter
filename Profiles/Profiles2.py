import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import imageio


main_path="/home/oiangu/Desktop/Conical_Refraction_Polarimeter/Profiles/test4/"

path_dict={}
for (dirpath, dirnames, filenames) in os.walk(main_path):
    path_dict[dirpath]=filenames


images={}
for dirpath, filenames in path_dict.items():
    for filename in filenames:
        if 'PROFILES' not in filename:
            img = cv2.imread(dirpath+'/'+filename, cv2.IMREAD_ANYDEPTH)
            if img is not None:
                images[filename] = img
                #np.array(Image.open(image_path))
                #Image.open(image_path).show()
            else:
                logging.warning(f" Unable to import image {dirpath+'/'+filename}")
if len(images.values())==0:
    # Then no valid images were introduced!
    logging.error(" No valid images were selected!")


for dirpath, filenames in path_dict.items():
    os.makedirs(f"{dirpath}/temp/",exist_ok=True)
    for filename in filenames:
        if 'PROFILES' not in filename:
            prof_x=np.sum(images[filename], axis=0)
            prof_y=np.sum(images[filename], axis=1)
            fig = plt.figure(figsize=(2*6, 2*6))
            axes=fig.subplots(2,2)
            cm=axes[0, 0].imshow(images[filename], cmap='viridis')
            axes[0,0].grid(True)
            axes[0,1].scatter(prof_y, np.arange(len(prof_y)), s=1, label=f'Intensity profile in y')
            axes[0,1].set_ylim((0,len(prof_y)))
            axes[0,1].invert_yaxis()
            axes[1,0].scatter(np.arange(len(prof_x)), prof_x, s=1, label=f'Intensity profile in y')
            axes[1,0].set_xlim((0,len(prof_x)))
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
            X,Y = np.meshgrid(np.arange(len(prof_x)),np.arange(len(prof_y)))
            im=images[filename].transpose()
            fig.suptitle(f"Intesity Profiles for Image\n{filename}")
            files_for_gif=[]
            cbax=fig.add_axes([0.54,0.05,0.4,0.01])
            fig.colorbar(cm, ax=axes[0,0], cax=cbax, orientation='horizontal')
            for theta in np.linspace(0, 360, 50):
                ax.clear()
                ax.plot_surface(X, Y, im, rcount=len(prof_y), ccount=len(prof_x), cmap='viridis') # rstride=1, cstride=1, linewidth=0
                cset = ax.contourf(X, Y, im, 2, zdir='z', offset=-0.078*np.max(im), cmap='viridis', alpha=0.5)
                cset = ax.contourf(X, Y, im, 1, zdir='x', offset=-8, cmap='viridis')
                cset = ax.contourf(X, Y, im, 1, zdir='y', offset=0, cmap='viridis')
                ax.set_xlabel('Y')
                #ax.set_xlim(-8, 8)
                ax.set_ylabel('Z')
                #ax.set_ylim(-10, 8)
                ax.set_zlabel('Intensity')
                ax.set_zlim(-0.078*np.max(im), np.max(im))
                ax.set_title("Image intensity 3D plot")
                ax.view_init(10, theta)
                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.3, 1.3, 1.3, 1]))
                plt.savefig(f"{dirpath}/temp/PROFILES_theta_{theta}_{filename}")
                files_for_gif.append(f"{dirpath}/temp/PROFILES_theta_{theta}_{filename}")

            with imageio.get_writer(f"{dirpath}/PROFILES_{filename[:-4]}.gif", mode='I') as writer:
                for frame in files_for_gif:
                    image = imageio.imread(frame)
                    writer.append_data(image)
                    os.remove(frame)
    os.rmdir(f"{dirpath}/temp/")
