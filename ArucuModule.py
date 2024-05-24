import cv2
import cv2.aruco as aruco
import numpy as np
import os

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
