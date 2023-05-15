# 拍摄图片命名为test_for_cross.jpg,存入img文件夹中

import cv2

cap = cv2.VideoCapture(0)
# 获取图片
ret, frame = cap.read()
# 保存图片
cv2.imwrite('img/test_for_cross_3.jpg', frame)
# 释放资源
cap.release()
