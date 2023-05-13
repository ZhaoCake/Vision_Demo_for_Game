import cv2
from pyzbar.pyzbar import decode

# 打开摄像头
cap = cv2.VideoCapture(0)
# 设置摄像头分辨率320*240
cap.set(3, 160)
cap.set(4, 120)

while True:
    # 读取摄像头中的帧
    ret, frame = cap.read()
    # frame = frame.resize(240, 320)

    # 解码帧中的二维码
    decoded_objects = decode(frame)

    # 显示解码后的结果
    for obj in decoded_objects:
        print("Link:", obj.data.decode())
        cv2.putText(frame, str(obj.data.decode()), (obj.rect.left, obj.rect.top), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

    # 显示摄像头中的图像
    cv2.imshow("Camera", frame)

    # 检查是否按下了q键，如果是则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
