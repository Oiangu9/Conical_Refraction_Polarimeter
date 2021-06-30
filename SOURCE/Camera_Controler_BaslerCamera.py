from time import sleep
import cv2
import sys
import os
import numpy as np
import datetime
from pypylon import pylon
from pypylon import genicam

class Camera_Controler:
    def __init__(self, angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar):
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

    def test_Camera(self):
        pass
    def grab_and_fix_reference(self):
        pass
    def take_and_process_frames(self, num_frames, save_every):
        pass


class Basler_Camera(Camera_Controler):
    """
    Our model:
        https://docs.baslerweb.com/aca720-520um

    """
    def __init__(self, angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar, width, height, offsetX, offsetY, maxBufferNum, mainThreadPlotter, previs_ms):
        Camera_Controler.__init__(self,angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar)
        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        # Print the model name of the camera.
        print("Using Basler Camera: ", self.camera.GetDeviceInfo().GetModelName())
        # The parameter MaxNumBuffer can be used to control the count of buffers
        # allocated for grabbing. The default value of this parameter is 10.
        self.camera.MaxNumBuffer = maxBufferNum

        self.w=width
        self.h=height

        self.camera.Width.SetValue(width)
        self.camera.Height.SetValue(height)
        self.camera.OffsetX.SetValue(offsetX)
        self.camera.OffsetY.SetValue(offsetY) #relative to the pixel 0,0 of the camera sensor

        self.images=np.zeros((images_chunk,height, width), dtype=np.uint16)
        self.names=['i' for i in range(images_chunk)]
        self.previs_ms=previs_ms
        self.mainThreadPlotter=mainThreadPlotter
        self.camera.Close(  )


    def test_Camera(self):
        self.camera.Open() # we open the camera
        # The camera device is parameterized with a default configuration which
        # sets up free-running continuous acquisition.
        self.camera.StartGrabbing()
        im=0
        while(self.stop_camera==False):
            im+=1
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Show the image data.
                img = grabResult.Array # they are type uint8
                self.mainThreadPlotter.emit(img, self.previs_ms, f"Grabbed test image #{im}")
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
        self.camera.StopGrabbing()
        self.camera.Close()


    def grab_and_fix_reference(self):
        self.camera.Open()
        # Start the grabbing of chunk images.
        # The camera device is parameterized with a default configuration which
        # sets up free-running continuous acquisition.
        self.camera.StartGrabbingMax(self.images_chunk)
        # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        # when c_countOfImagesToGrab images have been retrieved.
        im=-1
        while self.camera.IsGrabbing():
            im+=1
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Access the image data.
                self.images[im,:,:]=grabResult.Array
                self.names[im]=(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
        self.camera.Close()

        # Process the captured images
        self.image_manager.input_raw_images( self.images.astype(np.float64)/np.expand_dims(np.amax(self.images, axis=(1,2)), (1,2)), self.names)
        self.image_manager.compute_raw_to_iX()

        # Get angles
        self.angle_algorithm.reInitialize(self.image_manager)
        self.compute_angles_func()
        # Set their average angle as the reference angle given the custom 'zero'
        self.angle_algorithm.set_reference_angle(self.ref_angle)
        self.angle_algorithm.process_obtained_angles()
        # Show results (and save them if asked by user)
        self.image_manager.plot_rings_and_angles(self.angle_algorithm.polarization, self.angle_algorithm.polarization_precision, output_path=self.reference_path)


    def take_and_process_frames(self, num_frames, save_every):
        total_chunks=num_frames//self.images_chunk+1
        self.camera.Open()
        # Start the grabbing of chunk images.
        # The camera device is parameterized with a default configuration which
        # sets up free-running continuous acquisition.
        for chunk in range(total_chunks):
            self.camera.StartGrabbingMax(self.images_chunk)
            # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
            # when c_countOfImagesToGrab images have been retrieved.
            im=-1
            while self.camera.IsGrabbing():
                im+=1
                # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                # Image grabbed successfully?
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    self.images[im,:,:]=grabResult.Array
                    self.names[im]=(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                grabResult.Release()

            # Process the captured images
            self.image_manager.input_raw_images(self.images.astype(np.float64)/np.expand_dims(np.amax(self.images, axis=(1,2)), (1,2)), self.names)
            self.image_manager.compute_raw_to_iX()
            # Get angles
            self.angle_algorithm.reInitialize(self.image_manager)
            self.compute_angles_func()
            self.angle_algorithm.process_obtained_angles()

            # Show results (and save them if chunk%outputEvery==0)
            self.image_manager.plot_rings_and_angles(self.angle_algorithm.polarization, self.angle_algorithm.polarization_precision,
                output_path=None if chunk%save_every!=0 else self.output_path)

            # Update progressBar
            self.progressBar.emit(100*chunk/total_chunks)

            # Check if Stop was hit by user
            if self.stop_camera:
                self.stop_camera=False
                self.progressBar.emit(0)
                self.camera.Close()
                return 1

        self.camera.Close()
        self.progressBar.emit(100)
