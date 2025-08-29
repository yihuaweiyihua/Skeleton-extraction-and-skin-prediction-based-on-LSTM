# Skeleton-extraction-and-skin-prediction-based-on-LSTM

## 项目来源概述

本项目来自第五届计图人工智能挑战赛，主要任务：
1. **骨骼预测**：从3D人体点云数据中预测52个关键关节点的3D坐标
2. **蒙皮权重预测**：为每个顶点计算到各个关节的蒙皮权重，实现骨骼动画控制

## 解决思路

采用两阶段预测架构：
- **第一阶段**：使用PCT (Point Cloud Transformer) 模型预测骨骼关节位置
- **第二阶段**：基于预测的骨骼和顶点信息，使用GCN网络预测蒙皮权重

## 环境配置

### 系统要求
- Ubuntu 22.04.5 LTS
- CUDA 11.8

### 安装步骤

1. 创建conda环境：
conda create -n jittor python=3.9
conda activate jittor

2. 安装系统依赖：
sudo apt install python-dev libomp-dev

3. 安装Jittor框架：
python -m pip install jittor
python -m jittor.test.test_example

4. 安装其他依赖包：
pip install astunparse autograd cupy numpy pandas Pillow PyMetis six pyparsing scipy setuptools sympy tqdm einops huggingface_hub

5. 安装JittorGeometric：
git clone https://github.com/AlgRUC/JittorGeometric.git
cd JittorGeometric
pip install .
pip install -r requirements.txt

## 运行步骤
训练骨骼预测模型
python train_skeleton.py
训练蒙皮权重预测模型
python train_skin.py

使用模型进行预测：
骨骼预测
python predict_skeleton.py
蒙皮权重预测
python predict_skin.py

## 数据说明

- `Data/`：原始数据集
- `Output/`：训练好的模型
- `predict/`：测试数据集上的运行结果




