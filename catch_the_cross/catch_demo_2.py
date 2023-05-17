import cv2
import numpy as np


def hough_cross_detection(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
        # 拟合横向直线
        horizontal_lines = []
        for line in lines:
            rho, theta = line[0]
            if np.pi / 4 < theta < 3 * np.pi / 4:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                horizontal_lines.append([x1, y1, x2, y2])

        # 拟合纵向直线
        vertical_lines = []
        for line in lines:
            rho, theta = line[0]
            if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                vertical_lines.append([x1, y1, x2, y2])

        # 进行直线拟合
        horizontal_pts = np.array(horizontal_lines).reshape((-1, 1, 2))
        horizontal_line_params = cv2.fitLine(horizontal_pts, cv2.DIST_L2, 0, 0.01, 0.01)
        k1, b1 = horizontal_line_params[1] / horizontal_line_params[0], horizontal_line_params[3] - \
                 horizontal_line_params[2] * (horizontal_line_params[1] / horizontal_line_params[0])
        vertical_pts = np.array(vertical_lines).reshape((-1, 1, 2))
        vertical_line_params = cv2.fitLine(vertical_pts, cv2.DIST_L2, 0, 0.01, 0.01)
        k2, b2 = vertical_line_params[1] / vertical_line_params[0], vertical_line_params[3] - vertical_line_params[
            2] * (vertical_line_params[1] / vertical_line_params[0])

        # # 绘制拟合直线
        # 绘制横向直线
        x1, y1 = 0, int(b1)
        x2, y2 = gray.shape[1] - 1, int(k1 * (gray.shape[1] - 1) + b1)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制纵向直线
        x1, y1 = 0, int(b2)
        x2, y2 = gray.shape[1] - 1, int(k2 * (gray.shape[1] - 1) + b2)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 计算交点并绘制
        x0, y0 = int((b2 - b1) / (k1 - k2)), int(k1 * (b2 - b1) / (k1 - k2) + b1)
        cv2.circle(image, (x0, y0), 5, (0, 0, 255), -1)

        return image

    except Exception as e:
        print("Error:", e)
        exit()


# 测试代码
if __name__ == "__main__":
    path = "../img/test_for_cross.jpg"
    image = cv2.imread(path)
    result = hough_cross_detection(image)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
