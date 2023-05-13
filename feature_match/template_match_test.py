# 使用模板匹配的方式匹配图像

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img/20.jpg', 0)
img2 = img.copy()  # 用于绘制矩形

template = cv2.imread('img/template_1.jpg.png', 0)  # 模板图像
w, h = template.shape[::-1]  # 模板图像的宽和高

# 6种不同的匹配方法
# cv2.TM_CCOEFF # 相关系数匹配方法
# cv2.TM_CCOEFF_NORMED # 归一化相关系数匹配方法
# cv2.TM_CCORR # 相关匹配方法
# cv2.TM_CCORR_NORMED # 归一化相关匹配方法
# cv2.TM_SQDIFF # 差值平方匹配方法
# cv2.TM_SQDIFF_NORMED # 归一化差值平方匹配方法
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()

    # 匹配方法的真值
    method = eval(meth)

    # 模板匹配
    res = cv2.matchTemplate(img, template, method)

    # 获取最大值和最小值的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc

    # 否则取最大值
    else:
        top_left = max_loc

    # 右下角的坐标
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # 显示图像
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

# 效果差爆
