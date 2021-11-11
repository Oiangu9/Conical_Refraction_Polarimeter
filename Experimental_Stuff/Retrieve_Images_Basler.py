# ===============================================================================
#    This sample illustrates how to grab and process images using the CInstantCamera class.
#    The images are grabbed and processed asynchronously, i.e.,
#    while the application is processing a buffer, the acquisition of the next buffer is done
#    in parallel.
#
#    The CInstantCamera class uses a pool of buffers to retrieve image data
#    from the camera device. Once a buffer is filled and ready,
#    the buffer can be retrieved from the camera object for processing. The buffer
#    and additional image data are collected in a grab result. The grab result is
#    held by a smart pointer after retrieval. The buffer is automatically reused
#    when explicitly released or when the smart pointer object is destroyed.
# ===============================================================================
from pypylon import pylon
from pypylon import genicam
import cv2
import sys

# Number of images to be grabbed.
countOfImagesToGrab = 100000000000
one_every=40
photos_to_save=200
j=0

output_path="/home/melanie/Desktop/Conical_Refraction_Polarimeter/Experimental_Stuff/Fotos_Turpin/Day3/90/"
exp_name="90_"

# The exit code of the sample application.
exitCode = 0

try:
    # Create an instant camera object with the camera device found first.
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Print the model name of the camera.
    print("Using device ", camera.GetDeviceInfo().GetModelName())

    # demonstrate some feature access
    new_width = camera.Width.GetValue() - camera.Width.GetInc()
    if new_width >= camera.Width.GetMin():
        camera.Width.SetValue(new_width)

    # The parameter MaxNumBuffer can be used to control the count of buffers
    # allocated for grabbing. The default value of this parameter is 10.
    camera.MaxNumBuffer = 1
    camera.ExposureTime.SetValue(681)

    # Start the grabbing of c_countOfImagesToGrab images.
    # The camera device is parameterized with a default configuration which
    # sets up free-running continuous acquisition.
    camera.StartGrabbingMax(countOfImagesToGrab)
    center_1w=308
    center_1h=196
    center_2h=386
    center_2w=402
    w=150
    # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
    # when c_countOfImagesToGrab images have been retrieved.
    i=0
    while camera.IsGrabbing():
        i+=1
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            w0=grabResult.Width
            h0=grabResult.Height
            #print("SizeX: ", grabResult.Width)
            #print("SizeY: ", grabResult.Height)
            img = grabResult.Array
            cv2.imshow("{i}", img)
            cv2.waitKey(2)
            if i%one_every==0:
                print(43*'\n')
                print(f"Std Left {img[(center_1h-w):(center_1h+w),(center_1w-w):(center_1w+w)].std()}    \nMean Left {img[(center_1h-w):(center_1h+w),(center_1w-w):(center_1w+w)].mean()}    \nSum Left {img[(center_1h-w):(center_1h+w),(center_1w-w):(center_1w+w)].sum()}    ")
                print(f"Sum Full Image {img.sum()} ")
                cv2.imwrite(f"{output_path}/{exp_name}_{j}.png", img)
                j+=1
                if j>=photos_to_save:
                    break
            #cv2.destroyAllWindows()

            #cv2.imwrite("i_Custoom.png", img)
            #print("Gray value of first pixel: ", img[0, 0], "Image size: ", img.shape)
            #print(img[])
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    camera.Close()

except genicam.GenericException as e:
    # Error handling.
    print("An exception occurred.")
    print(e.GetDescription())
    exitCode = 1

sys.exit(exitCode)
