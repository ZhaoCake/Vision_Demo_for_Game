# 为了明确颜色过滤的上下限，我们需要一个小工具来获取颜色的上下限。
# 我们通过读取图片获得颜色的上下限，这样就可以在程序中直接使用了。

import cv2
import numpy as np

# 读取图片
path = "../img/object_color.png"
image = cv2.imread(path)

# 转换为hsv空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 获取全部像素点中的最大值和最小值作为颜色的上下限
lower_red = np.array([hsv[:, :, 0].min(), hsv[:, :, 1].min(), hsv[:, :, 2].min()])
upper_red = np.array([hsv[:, :, 0].max(), hsv[:, :, 1].max(), hsv[:, :, 2].max()])
print("lower_red:", lower_red)
print("upper_red:", upper_red)
