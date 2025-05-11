import cv2
import numpy as np
from openvino.runtime import Core

# 初始化
core = Core()

# 加载模型（示例使用face-detection）
xml_path = r"C:\Users\Administrator\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.9\models\intel\face-detection-0200\FP32\face-detection-0200.xml" 
bin_path = r"C:\Users\Administrator\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.9\models\intel\face-detection-0200\FP32\face-detection-0200.bin"
model = core.read_model(xml_path,bin_path)
compiled_model = core.compile_model(model, "CPU")


# 推理测试
image = cv2.imread("dataset/test/images/image3.jpg")    #替换测试图片
resize_image = cv2.resize(image,(256,256))
input_array = np.transpose(resize_image, (2, 0, 1))
input_tensor = np.expand_dims(input_array, 0)
input_rensor = input_tensor.astype(np.float32)
# 添加batch维度
print("input tensor shape:", input_tensor.shape)
results = compiled_model(input_tensor)[0]
print(results.shape)          # 应输出[1, 1, N, 7]
