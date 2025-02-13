import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import functional as F


def channel():
    image = cv2.imread('./images/red.png', cv2.IMREAD_COLOR)
    # BGR
    print(image[0][0])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # RGB
    print(rgb[0][0])
    # BGR2RGB 和 RGB2BGR 其实是等价的，= [:, :, ::-1]
    print(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[0][0])
    print(image[:, :, ::-1][0][0])
    # F.to_tensor 不改变排列
    print(F.to_tensor(image)[:, 0, 0])
    

def rotate():
    image = cv2.imread('./images/Maybach.png')
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 创建旋转矩阵
    # angel 逆时针旋转角度
    rotate_matrix = cv2.getRotationMatrix2D(center, angle=10, scale=1.0)
    rotate_image = cv2.warpAffine(image, rotate_matrix, (w, h))
    cv2.imwrite('./images/rotate.png', rotate_image)
    
    theta = 90
    radian = math.radians(theta)
    x, y = 0, 0
    cx, cy = 100, 100
    rx = cx + (x - cx) * math.cos(radian) + (y - cy) * math.sin(radian)
    ry = cy - (x - cx) * math.sin(radian) + (y - cy) * math.cos(radian)
    print(rx, ry)


def read_show():
    # 读取图片
    image_path = "./images/1.png"

    # cv2.IMREAD_COLOR      以彩色模式读取图像，忽略图像的透明度通道
    # cv2.IMREAD_GRAYSCALE  以灰度模式读取图像，shape = h, w
    # cv2.IMREAD_UNCHANGED  以原始模式读取图像，包括透明度通道（如果有）
    # cv2 读取图片时会把 RGB 的排列保存为 BGR 排列
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)    # 路径不能有中文，否则image为None   默认的是BGR排列

    # 写入时也会把 BGR 存储的图片还原为 RGB 排列保存
    cv2.imwrite("./images/1_copy.png", image)

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./images/1_BGR.png", image_RGB)
    image_BGR = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./images/1_RGB.png", image_BGR)

    assert image is not None, f"failed to read image: {image_path}"
    height, width, channels = image.shape
    print(type(image))                      # numpy.ndarray
    print(f'image shape: {image.shape}')    # h, w, c
    print(f"图像尺寸：{width} x {height}")
    print(f"通道数：{channels}")

    # PIL 图像（Image.Image类型） | numpy.ndarray → torch.Tensor
    # 1. h, w, c → c, h, w
    # 2. 数据范围归一化到 [0.0, 1.0]
    image0 = F.to_tensor(image)
    print(image0.shape)
    print(image0.dtype) # float32
    print(image0[0][:10][:10])

    # 显示图片
    cv2.imshow("IMG", image)            # 窗口名，imread读入的图像 cv2 显示也是按照BGR排列
    cv2.waitKey(0)                      # 等待键盘输入，单位为毫秒，0表示无限等待
    cv2.destroyAllWindows()             # 销毁所有窗口

    plt.subplot(1,1,1)
    plt.imshow(image)                   # plt则是安装RGB排列
    plt.axis('off')
    plt.title('bgr')
    plt.show()

    # 图像颜色空间转换
    # cv2.COLOR_RGB2GRAY    灰度化：彩色图像转为灰度图像    通道 3 → 1      丢失彩色
    # cv2.COLOR_GRAY2RGB    彩色化：灰度图像转为彩色图像    通道 1 → 3      增加了 “多余” 的信息。并且这种转换并不能真正恢复出原始的彩色图像，只是生成了一个看起来是彩色但实际上三个通道值完全相同的伪彩色图像。
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'image shape: {image.shape}')
    cv2.imshow("IMG", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.subplot(1,1,1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('bgr')
    plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print(f'image shape: {image.shape}')
    cv2.imshow("IMG", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("IMG", torch.randint(0, 256, (300, 400, 3)).numpy().astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rotate()
