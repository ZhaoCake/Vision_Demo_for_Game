import cv2
import numpy as np


class ColorDetector:
    def __init__(self, color):
        self.color = color
        self.lower, self.upper = 0, 0
        self.x = 0

    def get_hsv_range(self):
        if self.color == "red":
            self.lower = np.array([0, 43, 46])
            self.upper = np.array([7, 255, 255])
        elif self.color == "green":
            self.lower = np.array([33, 43, 46])
            self.upper = np.array([75, 255, 255])
        elif self.color == "blue":
            self.lower = np.array([110, 43, 46])
            self.upper = np.array([119, 255, 255])
        return 0

    def process_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.get_hsv_range()
        mask = cv2.inRange(hsv_image, self.lower, self.upper)
        return mask

    def draw_center(self, frame, mask):
        M = cv2.moments(mask)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            print(f"Center of all white regions: ({x}, {y})")
            cv2.arrowedLine(frame, (x - 10, y), (x + 10, y), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (x, y - 10), (x, y + 10), (0, 255, 0), 2)
        else:
            print("No white regions found in the image.")
            x = 0

        return x


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    color_aim = input("Please input the color you want to detect: ")
    color_detector = ColorDetector(color_aim)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = color_detector.process_image(frame)
        color_detector.draw_center(frame, mask)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        if cv2.waitKey(1) == ord('q'):
            break
