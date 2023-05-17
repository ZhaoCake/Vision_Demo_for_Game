import cv2
import numpy as np
from pyzbar.pyzbar import decode


def get_hsv_range(color):
    if color == "red":
        lower = np.array([0, 43, 46])
        upper = np.array([7, 255, 255])
    elif color == "green":
        lower = np.array([33, 43, 46])
        upper = np.array([75, 255, 255])
    elif color == "blue":
        lower = np.array([110, 43, 46])
        upper = np.array([119, 255, 255])
    else:
        print("Wrong color input!")
        exit()
    return lower, upper


def process_image(image, color):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = get_hsv_range(color)
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask


def draw_center(frame, mask):
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


def qr_code_detect():
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率320*240
    cap.set(3, 160)
    cap.set(4, 120)
    while True:
        # 读取摄像头中的帧
        ret, frame = cap.read()
        # 解码帧中的二维码
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            print(obj.data.decode())
            if obj.data.decode():
                cap.release()
                return obj.data.decode()


def color_catch(color):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = process_image(frame, color)
        x = draw_center(frame, mask)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.waitKey(100)
        # 如果x在图像中心附近，结束
        if 50 < x < 90:
            cap.release()
            cv2.destroyAllWindows()
            return True


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
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
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
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
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


def cross_correct():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = hough_cross_detection(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(200)


def main():
    input("Press Enter to start")
    color_list = qr_code_detect()
    color_list = color_list.split(',')
    print(color_list)
    for color in color_list:
        if color_catch(color):
            print("catch it")
        input("Press Enter to correct")
        # 由于材料限制，特加入一个跳过纠偏的功能
        if input("Press Enter to correct") == 'n':
            continue
    print("All done!")


if __name__ == '__main__':
    '''
    1、接受指令，进行二维码扫描，并将读取到的结果（3种颜色）返回到color_list中
    2、接受主控板抓取信息，每0.2秒识别一次，判断是否抓起，并返回抓信息
    3、接受主控板纠偏指令，进行纠偏(直到十字与十字重合，偏差在一定范围内，返回纠偏完成信息)
    4、接受主控板抓取指令，进行识别（重复2、3）
    5、全部3种颜色内容完成，收工
    '''
    main()
