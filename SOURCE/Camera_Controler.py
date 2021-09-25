from time import sleep
import cv2
import sys
import os
import numpy as np
import datetime
from picamera import PiCamera
import picamera.array
from pypylon import pylon
from pypylon import genicam

class Camera_Controler:
    def __init__(self, angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar, deg_or_rad):
        self.angle_algorithm=angle_algorithm
        self.compute_angles_func=compute_angles_func
        self.ref_angle=ref_angle
        self.images_chunk=images_chunk
        self.image_manager=image_manager
        self.stop_camera=False
        self.save_outputs=save_outputs
        self.output_path=None
        self.reference_path=None
        if save_outputs:
            os.makedirs(f"{output_path}/Life_Take/Reference/", exist_ok=True)
            os.makedirs(f"{output_path}/Life_Take/Sequence/", exist_ok=True)
            self.output_path=output_path+"/Life_Take/Sequence/"
            self.reference_path=output_path+"/Life_Take/Reference/"
        self.progressBar=progressBar
        self.deg_or_rad=deg_or_rad #0 is rad, 1 is deg
        self.unit='deg' if deg_or_rad else 'rad'

    def test_Camera(self):
        pass
    def grab_and_fix_reference(self):
        pass
    def take_process_frames(self, num_frames, save_every):
        pass
