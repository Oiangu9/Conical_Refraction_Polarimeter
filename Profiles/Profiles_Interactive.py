import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#image_full_path="/home/oiangu/Desktop/Conical_Refraction_Polarimeter/DATA/SIMULATED/SIMULATED_Utukuri_Examples/2202_Resolution/[Big_Radious_Very_Thin_Width_]__PolAngle_-1.196600000000000_CRAngle_-2.393200000000000_Z_0_R0_30.25_R0_pix_475.58.png"
#image_full_path="/home/oiangu/Desktop/Conical_Refraction_Polarimeter/DATA/SIMULATED/SIMULATED_Utukuri_Examples/1101_Resolution/[Big_Radious_Very_Thin_Width_]__PolAngle_-1.196600000000000_CRAngle_-2.393200000000000_Z_0_R0_30.25_R0_pix_237.79.png"
image_full_path="/home/oiangu/Desktop/Conical_Refraction_Polarimeter/Experimental_Stuff/Fotos_Turpin/Haz_sin_mas_ajustr_gaussiana.png"
im = cv2.imread(image_full_path, cv2.IMREAD_ANYDEPTH)
if im is None:
    print(f" Unable to import image {image_full_path}")
    raise ValueError

plot3d_resolution=0.6
prof_x=np.sum(im, axis=0)
prof_y=np.sum(im, axis=1)
fig = plt.figure(figsize=(2*6, 2*6))
axes=fig.subplots(2,2)
cm=axes[0, 0].imshow(im, cmap='viridis')
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
X,Y = np.meshgrid(np.arange(len(prof_y)),np.arange(len(prof_x)))
fig.suptitle(f"Intesity Profiles for Image\n{image_full_path.split('/')[-1]}")
files_for_gif=[]
cbax=fig.add_axes([0.54,0.05,0.4,0.01])
fig.colorbar(cm, ax=axes[0,0], cax=cbax, orientation='horizontal')
theta=25
phi=30
ax.plot_surface(X, Y, im.T, rcount=int(len(prof_y)*plot3d_resolution), ccount=int(len(prof_x)*plot3d_resolution), cmap='viridis') # rstride=1, cstride=1, linewidth=0
#cset = ax.contourf(X, Y, im, 2, zdir='z', offset=-20, cmap='viridis', alpha=0.5)
#cset = ax.contourf(X, Y, im, 1, zdir='x', offset=-8, cmap='viridis')
#cset = ax.contourf(X, Y, im, 1, zdir='y', offset=0, cmap='viridis')
ax.set_xlabel('Y')
#ax.set_xlim(-8, 8)
ax.set_ylabel('X')
#ax.set_ylim(-10, 8)
ax.set_zlabel('Intensity')
ax.set_zlim(-0.078*np.max(im), np.max(im))
ax.set_title("Image intensity 3D plot")
ax.view_init(10, theta)
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.3, 1.3, 1.3, 1]))
plt.show()
