# 将orb_test_2.py中代码改写用于摄像头0的实时匹配
# 匹配的目标图像为../img/0.jpg
# 被匹配的图像为摄像头0的实时图像
# 使用sift特征检测器

import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('./img/template_1.png')
# img1 = cv2.convertScaleAbs(img1) # 为了显示图像，将图像转为uint8类型

# sift特征检测器
sift = cv2.xfeatures2d.SIFT_create()

# 检测关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)

# 暴力匹配
bf = cv2.BFMatcher()

# 创建VideoCapture对象
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图像
    ret, img2 = cap.read()

    # 转为灰度图像
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 检测关键点和描述符
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 匹配描述符
    matches = bf.knnMatch(des1, des2, k=2)  # 返回k个最佳匹配

    # 获取在第二张图像中匹配到的关键点的坐标
    keypoints2 = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # 计算第一张图像中前十个关键点的最小外接矩形
    rect = cv2.minAreaRect(keypoints2)
    # 将最小外接矩形转换为矩形框的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # 在图像中绘制矩形框
    img_with_box = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)

    # 按照距离排序
    matches = sorted(matches, key=lambda x: x[0].distance)

    # 绘制前10个匹配
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    # 缩小图像
    img3 = cv2.resize(img3, (0, 0), fx=0.6, fy=0.6)

    # 用cv2.imshow()显示图像
    cv2.imshow('img3', img3)

    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
