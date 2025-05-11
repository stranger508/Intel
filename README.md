# Intel_openvino
# YOLO目标检测模型部署实践（YOLOv5 / YOLOv8 + OpenVINO）

本项目完整实现了从图像标注、模型训练、格式转换，到使用 Intel OpenVINO 工具进行模型部署与推理的全过程。通过 YOLOv5 或 YOLOv8 进行目标检测模型训练，再使用 OpenVINO Model Optimizer 将模型转换为高效的 IR 格式进行推理部署。

## 项目亮点
- 支持 Pascal VOC 格式标注与 YOLO 标签转换
- 支持 YOLOv5 / YOLOv8 模型训练
- 跨平台适配经验：从 Mac (ARM) 到 Windows (x64)
- 使用 OpenVINO 实现部署优化，显著提升推理速度
- 处理了多个模型转换与运行时错误，过程详尽可复现
  
## 技术栈
- Python
- YOLOv5 / YOLOv8
- OpenVINO Toolkit
- ONNX
- labelImg（标注工具）

在项目开发中，经历了数次环境不兼容与模型转换失败的问题，特别是在 Mac ARM 架构下，OpenVINO 存在大量依赖冲突。最终转战 Windows 平台，查阅了大量官方资料、GitHub 项目和学术论文，解决了模型格式转换、输入输出维度匹配等核心问题，并成功部署运行了推理流程。项目虽小，但完整经历了从理论到实操的工程闭环，对目标检测部署有深刻理解。

