import cv2
import numpy as np


class ColorDetector:
    def __init__(self, r, g, b, threshold=10):
        self.color = np.uint8([[[b, g, r]]])  # 要求的颜色
        self.threshold = threshold  # 颜色相似度阈值

    # def absdiffHSV(self, img1, img2):
    #     # hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #     # hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    #     hsv1 = img1
    #     hsv2 = img2
    #     diff_h = np.abs(hsv1[:, :, 0] - hsv2[:, :, 0])
    #     diff_s = np.abs(hsv1[:, :, 1] - hsv2[:, :, 1])
    #     diff_v = np.abs(hsv1[:, :, 2] - hsv2[:, :, 2])
    #     diff_h = np.uint8(np.minimum(diff_h, 360 - diff_h)/2)
    #     return cv2.merge([diff_h, diff_s, diff_v])

    def process_image(self, image):
        # 将图像转换为HSV颜色空间，这样可以减小光线变化的影响
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 转换为HSV颜色空间, 便于计算颜色相似度
        # 计算颜色相似度
        color_image = np.zeros_like(image)  # 创建一个与原图像大小相同的图像
        color_image[:] = self.color  # 将要求的颜色填充到图像中
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV) # 将图像转换为HSV颜色空间
        diff = cv2.absdiff(image, color_image)  # 计算颜色相似度
        # diff = cv2.cvtColor(diff, cv2.COLOR_HSV2BGR) # 将图像转换为BGR颜色空间
        dist = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        # dist = cv2.equalizeHist(dist) # 直方图均衡化
        mask = cv2.inRange(dist, 0, self.threshold)  # 计算二值图像

        # 对图像进行二值化处理
        ret, thresholded = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # 对图像进行二值化处理

        # 返回处理后的图像
        return color_image, dist, thresholded


def draw_center(frame, processed_image):
    M = cv2.moments(processed_image)
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


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        # 从摄像头读取图像
        ret, frame = cap.read()

        # 如果图像读取失败，则退出循环
        if not ret:
            break

        # detector = ColorDetector(170, 97, 97, 20) # 红色对象
        # detector = ColorDetector(20, 20, 200, 2) # 蓝色对象
        detector = ColorDetector(210, 25, 25, 20)  # 红色物体

        # 处理图像
        p_color, p_diff, processed_image = detector.process_image(frame)

        # 显示中心点
        draw_center(frame, processed_image)

        # 显示原图像和处理后图像
        cv2.imshow('Original Image', frame)
        # # cv2.imshow('Color Image', p_color)
        cv2.imshow('Diff Image', p_diff)
        cv2.imshow('Proccess_', processed_image)

        # 按下ESC键退出循环
        if cv2.waitKey(1) == 27:
            break

    # 释放VideoCapture对象和窗口
    cap.release()
    cv2.destroyAllWindows()
