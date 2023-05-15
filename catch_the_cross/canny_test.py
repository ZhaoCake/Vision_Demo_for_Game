# 使用hsv空间下的canny检测查看效果

import cv2
import numpy as np


def canny_for_test():
    pass


if __name__ == "__main__":
    path = "../img/test_for_cross.jpg"
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([162, 83, 99])
    upper_red = np.array([176, 255, 178])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow("mask", mask)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
