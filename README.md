# 视觉部分代码（初版+）

## 说明

**注意**：此代码为初版，仅供参考，不保证能够正常运行，也不保证能够满足实际需求，仅供参考。

**注意**：目前作了一些改动，同时初版代码有许多错误的地方，暂时没有进行整理。

## 基本说明

- 此代码在ubuntu22.04上完成，python版本3.9.16，系统架构x86_64，尚未在arm64设备上运行过。
- 依赖文件requirements.txt在当前目录下，其中numpy、pyzbar、pyserial，通过收集终端测试，均在arm64系统架构上安装对应版本。
  尽管如此，依然存在由于库版本的不同导致需要对代码进行修改的情况。

## 任务说明

由于尚且不知晓具体的流程，根据群内之前视觉文件夹颜色识别代码，主要完成的视觉任务是颜色识别并返回应当的抓去顺序。
因此此份代码主要任务如下：

- 打开摄像头，读取二维码，获取二维码中信息，关闭摄像头
- 假设获取的信息是抓取物体颜色的顺序，那么这一步将保存这个信息
- 颜色识别，将目前的三个物体的中心坐标找到
- 按照任务所需要的顺序，将左中右三个位置物体的抓取顺序输出，通过串口通讯传回主控板

窥屏了解到还可能需要使用视觉进行位置矫正等任务，一并附上。

## 内容说明

### requirements.txt

项目依赖包，可通过如下命令安装

'''
pip install -r requirements.txt
'''

### color_recog.py

颜色识别、找出目标颜色区域中心。为其他文件所调用。
由于场地等的限制，可能出现由于光照变化造成的颜色识别误差, 若情况严重之后可以通过转换到HSV空间中减小光照条件变化导致的问题。
同时，直接运行这个代码可以进行对某颜色中心坐标的输出，如果之后确定了使用视觉进行抓取的纠正的话，那么可以将代码整合即可。

### version_demo_1.py

此部分代码完成上述主要任务。其中串口通讯部分需要根据情况修改。可以通过$ls\ /dev/tty*$等命令获取串口名。
颜色识别中，需要根据真实反映的颜色修改rgb与阈值以达到较好效果。

### 其他

#### version_demo_2.py

这是一份霍夫变换检测圆的代码，但是，它对算力要求比想象中的高的多。如果是这样的算力需求，还不如直接采用基于深度学习的目标检测算法框选物体计算圆心。
因此两种方案，一种是依然使用同找到区域圆心的方法，但是对于这种线条很细的多个圆环来说效果不见得好，需要实际的测试；第二种方案是使用目标检测，这种对象简单的任务，可以选择较小的轻量的网络，仅仅是推理的算力要求不比其他几种方案多，数据集也不需要很多，保守估计拍照30张左右也就够了，尝试成本不高，是可以试试的。

#### test_result.txt

使用红色与蓝色记号笔在摄像头面前测出的效果，没有绿色笔故有一项为零，绿色部分排在最左边。

#### QR_recog.py

对二维码的扫描测试所用，不必管它。

#### red_blue_green.png

生成的二维码，包含信息“red,blue,green”，可用于测试。