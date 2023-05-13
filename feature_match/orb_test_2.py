# 将orb_test.py中代码改写用于摄像头0的实时匹配
# 匹配的目标图像为../img/0.jpg
# 被匹配的图像为摄像头0的实时图像

import cv2

# 读取图像
img1 = cv2.imread('./img/0.jpg', 0)

# ORB特征检测器
orb = cv2.ORB_create()

# 检测关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)

# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 创建VideoCapture对象
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图像
    ret, img2 = cap.read()

    # 转为灰度图像
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 检测关键点和描述符
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 匹配描述符
    matches = bf.match(des1, des2)

    # 按照距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制前10个匹配
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    # 缩小图像
    img3 = cv2.resize(img3, (0, 0), fx=0.3, fy=0.3)

    # 用cv2.imshow()显示图像
    cv2.imshow('img3', img3)

    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
