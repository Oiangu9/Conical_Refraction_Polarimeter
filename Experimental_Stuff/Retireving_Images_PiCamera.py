from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.resolution = (2592, 1944)
camera.framerate = 15
camera.start_preview()
sleep(10)
camera.capture('/home/pi/Desktop/image.jpg')

camera.stop_preview()
