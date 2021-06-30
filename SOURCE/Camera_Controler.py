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
    def take_process_frames(self, num_frames, save_every):
        pass

class Pi_Camera(Camera_Controler):
    '''
    YUV captures
        https://picamera.readthedocs.io/en/release-1.10/recipes2.html
    Raw Bayer Capture

    '''
    def __init__(self, angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar, width, height):
        Camera_Controler.__init__(self,angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar)
        self.camera = PiCamera()
        self.camera.resolution = (width, height)
        self.camera.framerate = 15
        self.outputStream = picamera.array.PiYUVArray(self.camera)
        self.raw_w=(width + 31) // 32 * 32
        self.raw_h=(height + 15) // 16 * 16
        self.images=np.zeros((images_chunk,height, width), dtype=np.uint16)
        self.names=['i' for i in range(images_chunk)]

    def reInitialize(self, angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar, width, height):
        Camera_Controler.__init__(self,angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar)
        #self.camera = PiCamera() If camera initialized multiple times yields an error
        self.camera.resolution = (width, height)
        self.camera.framerate = 15
        self.outputStream = picamera.array.PiYUVArray(self.camera)
        self.raw_w=(width + 31) // 32 * 32
        self.raw_h=(height + 15) // 16 * 16
        self.images=np.zeros((images_chunk,height, width), dtype=np.uint16)
        self.names=['i' for i in range(images_chunk)]

    def test_Camera(self):
        self.camera.start_preview()
        while(self.stop_camera==False):
            sleep(3)
        self.camera.stop_preview()
        self.stop_camera=False

    def grab_and_fix_reference(self):
        self.outputStream.truncate(0)
        self.camera.start_preview()
        # Camera warm-up time
        sleep(2)
        for im in range(self.images_chunk):
            # capture raw image
            self.camera.capture(self.outputStream, 'yuv')
            # put it in the array for the captured images of this chunk. The array is recorded in uint8
            self.images[im,:,:] = self.outputStream.array[:,:,0] #[h, w, yuv3]->[h,w,y]
            # reset stream
            cv2.imwrite("11.png", self.outputStream.array[:,:,0])
            self.outputStream.truncate(0)
            # get a name for the image
            self.names[im]=(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))
        self.camera.stop_preview()

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
        self.progressBar.emit(0)
        self.outputStream.truncate(0)
        self.camera.start_preview()
        # Camera warm-up time
        sleep(2)
        self.camera.stop_preview()
        total_chunks=num_frames//self.images_chunk+1
        for chunk in range(total_chunks):
            for im in range(self.images_chunk):
                # capture raw image
                self.camera.capture(self.outputStream, 'yuv')
                # put it in the array for the captured images of this chunk
                self.images[im,:,:] = self.outputStream.array[:,:,0] #[h, w, yuv3]->[h,w,luminance]
                # reset stream
                self.outputStream.truncate(0)
                # get a name for the image
                self.names[im]=(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"))

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
                return 1
        self.progressBar.emit(100)


class Basler_Camera(Camera_Controler):
    """
    Our model:
        https://docs.baslerweb.com/aca720-520um

    """
    def __init__(self, angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar, width, height, offsetX, offsetY, mainThreadPlotter, previs_ms):
        Camera_Controler.__init__(self,angle_algorithm, compute_angles_func, ref_angle, images_chunk, image_manager, save_outputs, output_path, progressBar)
        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Print the model name of the camera.
        print("Using Basler Camera: ", camera.GetDeviceInfo().GetModelName())
        # The parameter MaxNumBuffer can be used to control the count of buffers
        # allocated for grabbing. The default value of this parameter is 10.
        self.camera.MaxNumBuffer = 5

        self.camera.Width.SetValue(720)
        self.camera.Height.SetValue(540)
        self.camera.OffsetX.SetValue(0)
        self.camera.OffsetY.SetValue(0) #relative to the pixel 0,0 of the camera sensor

        self.images=np.zeros((images_chunk,height, width), dtype=np.uint16)
        self.names=['i' for i in range(images_chunk)]
        self.previs_ms=previs_ms
        self.mainThreadPlotter=mainThreadPlotter


    def test_Camera(self):
        camera.Open() # we open the camera
        # The camera device is parameterized with a default configuration which
        # sets up free-running continuous acquisition.
        camera.StartGrabbing()
        i=0
        while(self.stop_camera==False):
            i+=1
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Show the image data.
                img = grabResult.Array # they are type uint8
                self.mainThreadPlotter.emit(img, self.previs_ms, f"Grabbed test image #{i}")

                cv2.imwrite(f"{i}_Custoom.png", img)

            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
        camera.StopGrabbing()
        camera.Close()


    def grab_and_fix_reference(self):
        pass
    def take_process_frames(self, num_frames, save_every):
        pass
