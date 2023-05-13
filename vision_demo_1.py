import time

import cv2
import numpy as np
from pyzbar.pyzbar import decode

import color_recog


# 连接主控板串口
# ser = serial.Serial("/dev/tty*", 115200, timeout=0.5) # 串口名称根据实际情况修改，波特率根据主控板设置修改
# 接受C语言程序发送的整形数字数据
# def recv_int():
#     data = ser.readline()
#     if data:
#         return int(data.decode().strip()) # 将接收到的数据转换为整形数字
#     else:
#         return None

# # 发送整形数字数据到C语言程序
# def send_int(data):
#     ser.write(str(data).encode()) # 将整形数字转换为字符串并发送到串口

# 读取二维码获得抓取任务，读取到信息后停止读取
def get_task():
    cap = cv2.VideoCapture(0)
    while True:
        # 读取摄像头中的帧
        ret, frame = cap.read()

        # 解码帧中的二维码
        decoded_objects = decode(frame)

        # 显示解码后的结果
        for obj in decoded_objects:
            print("Link:", obj.data.decode())
            return obj.data.decode()
            cap.release()


def pro_task(task):  # 处理任务，将任务字符串转换为检测颜色的顺序，此处仅仅是例子，因为不知道二维码的具体格式
    task_pre = task.split(',')  # 将任务字符串分割成列表
    # 此处本可以直接在main函数中编写，但为了代码的可拓展，将其单独写成一个函数
    return task_pre


def get_aim_center():
    # 三个颜色检测器
    red_detector = color_recog.ColorDetector(220, 10, 10, 40)
    blue_detector = color_recog.ColorDetector(10, 10, 220, 40)
    green_detector = color_recog.ColorDetector(10, 220, 10, 40)
    # 读取摄像头
    cap = cv2.VideoCapture(0)
    value_list = []
    while True:
        # 读取摄像头中的帧
        ret, frame = cap.read()
        # 获取三个颜色检测器的
        _, _, red_result = red_detector.process_image(frame)
        red_c = color_recog.draw_center(frame, red_result)
        _, _, blue_result = blue_detector.process_image(frame)
        blue_c = color_recog.draw_center(frame, blue_result)
        _, _, green_result = green_detector.process_image(frame)
        green_c = color_recog.draw_center(frame, green_result)
        # 将检测结果添加到列表中
        value_list.append([red_c, green_c, blue_c])  # 按照红绿蓝的顺序添加
        if len(value_list) > 10:
            break
    # 释放摄像头
    cap.release()
    value_list = np.array(value_list)
    value_list = value_list[np.sum(value_list, axis=1) != 0]
    value = np.mean(value_list, axis=0)
    value = value.tolist()
    value = dict(zip(['red', 'green', 'blue'], value))
    # 根据三个值的大小确定颜色的顺序，数值大的在左边，数值小的在右边
    if value['red'] > value['green'] and value['red'] > value['blue']:
        if value['green'] > value['blue']:
            return [0, 1, 2]
        else:
            return [0, 2, 1]
    elif value['green'] > value['red'] and value['green'] > value['blue']:
        if value['red'] > value['blue']:
            return [1, 0, 2]
        else:
            return [2, 0, 1]
    else:
        if value['red'] > value['green']:
            return [1, 2, 0]
        else:
            return [2, 1, 0]


def main():
    task = get_task()  # 读取二维码获得抓取任务，由于不知道到底是什么样的二维码，所以生成了一个输出“red,blue,green”的二维码替代
    task_list = pro_task(task)  # 将任务转换成检测颜色的顺序，将红绿蓝排序，例如检测任务为红蓝绿,返回的列表为['red', 'blue', 'green']
    time.sleep(3)  # 由于不知道二维码的读取时机，暂时就写在一起了
    center_list = dict(zip(['red', 'green', 'blue'], get_aim_center()))
    catch_list = []
    for i in task_list:
        catch_list.append(center_list[i])
        # 含义： |2|1|0| 三个位置，先后输出的值，表示先后抓取的位置
    print(catch_list)


if __name__ == "__main__":
    main()
