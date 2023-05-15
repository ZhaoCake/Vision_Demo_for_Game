import cv2
import numpy as np


def hough_cross_detection(image):
    try:
        # 图像预处理：hsv空间提取紫红色区域
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([162, 83, 99])
        upper_red = np.array([176, 255, 178])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)

        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)  # 第一个参数是一个二值化图像，所以在进行霍夫变换之前要首先进行二值化，或者进行 Canny 边缘检测。
        # 第二和第三个值分别代表 ρ 和 θ 的精确度。 第四个参数是阈值，只有累加其中的值高于阈值时才被认为是一条直线，也可以把它看成能 检测到的直线的最短长度（以像素点为单位）。

        # 提取直线的端点坐标
        points = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            points.append([(x1, y1), (x2, y2)])

        # 寻找十字交叉点
        cross_points = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                pt1_1, pt1_2 = points[i]
                pt2_1, pt2_2 = points[j]
                if abs(pt1_1[0] - pt1_2[0]) > 5 and abs(pt2_1[1] - pt2_2[1]) > 5:
                    # 计算两条直线的交点
                    x = (pt1_1[1] - pt2_1[1] + (pt2_1[0] - pt1_1[0]) * (pt1_1[1] - pt1_2[1]) / (pt1_1[0] - pt1_2[0]) - (
                                pt2_1[0] - pt1_1[0]) * (pt2_1[1] - pt2_2[1]) / (pt2_1[0] - pt2_2[0])) / (
                                    (pt2_1[1] - pt2_2[1]) / (pt2_1[0] - pt2_2[0]) - (pt1_1[1] - pt1_2[1]) / (
                                        pt1_1[0] - pt1_2[0]))
                    y = ((pt1_1[1] - pt1_2[1]) / (pt1_1[0] - pt1_2[0])) * (x - pt1_1[0]) + pt1_1[1]
                    cross_points.append((int(x), int(y)))

        # 在图像上标出十字交叉点
        for point in cross_points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)

        # 在图像上标出直线
        for point in points:
            cv2.line(image, point[0], point[1], (0, 0, 255), 2)

        return image

    except Exception as e:
        print("Error:", e)
        exit()


# 测试代码
if __name__ == "__main__":
    image_path = "../img/test_for_cross.jpg"
    image = cv2.imread(image_path)
    result = hough_cross_detection(image)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
