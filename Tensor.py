# import cv2
# import numpy as np
# #read picture
# image = cv2.imread("dataset/test/images/image3.jpg")    #原始形状：（H，W，C），如（500， 300， 3）
#
# resized = cv2.resize(image, (256, 256))  #调整尺寸到 256✖️256
# input_tensor = resized.transpose(2, 0, 1)       #从 HWC -> CHW，形状变为（3，256，256）
# input_tensor = np.expand_dims(input_tensor, 0)  #添加批次维度，最终形状（1，3，256，256）
# input_tensor = input_tensor.astype(np.float32) #确保数据类型正确


import cv2
import numpy as np

# 读取图片
image = cv2.imread("dataset/test/images/image3.jpg")  # 确保图片路径正确

# 检查图片是否加载成功
if image is None:
    print("错误：无法加载图像，请检查文件路径或文件是否存在！")
    exit()  # 退出程序

# 打印原始图像形状
print(f"原始图像形状 (H, W, C): {image.shape}")

# 预处理
resized = cv2.resize(image, (256, 256))  # 调整尺寸到 256x256
print(f"调整尺寸后的形状 (H, W, C): {resized.shape}")

input_tensor = resized.transpose(2, 0, 1)  # 从 HWC → CHW，形状变为 (3, 256, 256)
print(f"转置后的形状 (C, H, W): {input_tensor.shape}")

input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加批次维度，形状变为 (1, 3, 256, 256)
print(f"添加批次维度后的形状 (B, C, H, W): {input_tensor.shape}")

input_tensor = input_tensor.astype(np.float32)  # 确保数据类型正确
print(f"最终张量数据类型: {input_tensor.dtype}")

# 输出成功信息
print("预处理完成，张量已准备好输入模型！")