import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = frame

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 进行霍夫变换检测圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0,
                               maxRadius=0)

    # 筛选最大的圆并绘制轮廓
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x: x[2], reverse=True)  # 按半径大小排序
        x, y, r = circles[0]  # 选取最大的圆
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 255, 0), 3)

    # 显示结果图像
    cv2.imshow("Result", img)
    # 按下ESC键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放VideoCapture对象和窗口
cap.release()
cv2.destroyAllWindows()
