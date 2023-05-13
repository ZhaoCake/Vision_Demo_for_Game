# 使用速度更快的特征匹配的算法实现图像识别

import cv2
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('../img/test1.jpg', 0)
img2 = cv2.imread('../img/test4.jpg', 0)

# ORB特征检测器
orb = cv2.ORB_create()

# 检测关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配描述符
matches = bf.match(des1, des2)

# 按照距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制前10个匹配
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# 显示图像达到三分之一个屏幕的大小
plt.figure(figsize=(20, 20))

# 显示图像
plt.imshow(img3)
plt.show()
