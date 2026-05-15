import cv2

for i in range(10):
    cam = cv2.VideoCapture(i)
    print(i, cam.isOpened())
    cam.release()