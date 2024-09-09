import cv2
import matplotlib.pyplot as plt

# 读取图片
image_path = "./images/1.png"

# cv2.IMREAD_COLOR      以彩色模式读取图像，忽略图像的透明度通道
# cv2.IMREAD_GRAYSCALE  以灰度模式读取图像
# cv2.IMREAD_UNCHANGED  以原始模式读取图像，包括透明度通道（如果有）
image = cv2.imread(image_path, cv2.IMREAD_COLOR)    # 路径不能有中文，否则image为None   默认的是BGR排列

assert image is not None, f"failed to read image: {image_path}"
height, width, channels = image.shape
print(f'image shape: {image.shape}')    # h, w, c
print(f"图像尺寸：{width} x {height}")
print(f"通道数：{channels}")

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
