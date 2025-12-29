# 基于SAM3的点云实时抓取系统流程文档

## 1. 系统概述

本系统采用**眼在手外(Eye-to-Hand)**模式，结合SAM3语义分割模型与点云处理技术，实现基于自然语言提示的机器人智能抓取。

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户输入                                      │
│                    (抓取类别 Prompt)                                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              RealSense D435 相机                                     │
│         ┌──────────────┬──────────────┐                             │
│         │   RGB图像     │    深度图     │                             │
│         └──────┬───────┴──────┬───────┘                             │
└────────────────┼──────────────┼─────────────────────────────────────┘
                 │              │
                 ▼              │
┌────────────────────────┐      │
│     SAM3 模型推理       │      │
│  (RGB + Prompt → Mask) │      │
└───────────┬────────────┘      │
            │                   │
            ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Mask + 深度图 融合                                │
│                      ↓                                              │
│                  点云生成                                            │
│                      ↓                                              │
│              PCL 点云滤波处理                                        │
│                      ↓                                              │
│                抓取位姿估计                                          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    机械臂执行抓取                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 硬件配置

| 组件 | 型号/规格 | 用途 |
|------|----------|------|
| 深度相机 | Intel RealSense D435 | 获取RGB图像和深度信息 |
| 机械臂 | 根据实际配置 | 执行抓取动作 |
| 夹爪 | 二指平行夹爪 | 抓取目标物体 |
| 计算平台 | GPU工作站 | SAM3模型推理 |

### 1.3 软件依赖

```
- Python 3.8+
- PyTorch 2.0+
- SAM3 (Segment Anything Model 3)
- pyrealsense2
- Open3D / PCL
- NumPy, OpenCV
- 机械臂SDK (根据具体型号)
```

---

## 2. 详细流程步骤

### 2.1 步骤一：相机标定与手眼标定

#### 2.1.1 相机内参标定

```python
# 获取RealSense D435内参
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# 获取内参
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 内参矩阵 K
K = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])
```

#### 2.1.2 眼在手外标定

眼在手外模式下，相机固定在工作台上，需要标定相机坐标系到机械臂基座坐标系的变换矩阵 `T_cam_to_base`。

```
标定方法：
1. 在机械臂末端安装标定板
2. 控制机械臂移动到多个位姿（建议15-20个）
3. 记录每个位姿下：
   - 机械臂末端位姿 T_end_to_base
   - 相机检测到的标定板位姿 T_board_to_cam
4. 使用 AX=XB 求解器计算 T_cam_to_base
```

---

### 2.2 步骤二：实时图像采集

#### 2.2.1 RealSense D435 数据流获取

```python
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 配置数据流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 启动相机
        self.profile = self.pipeline.start(self.config)

        # 获取深度传感器并设置
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # 深度比例因子

        # 创建对齐对象（将深度图对齐到彩色图）
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        """获取对齐后的RGB图像和深度图"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()
```

#### 2.2.2 深度图预处理

```python
def preprocess_depth(depth_image, depth_scale, min_depth=0.1, max_depth=2.0):
    """
    深度图预处理
    Args:
        depth_image: 原始深度图 (uint16)
        depth_scale: 深度比例因子
        min_depth: 最小有效深度 (米)
        max_depth: 最大有效深度 (米)
    """
    # 转换为米制单位
    depth_meters = depth_image * depth_scale

    # 过滤无效深度值
    depth_meters[depth_meters < min_depth] = 0
    depth_meters[depth_meters > max_depth] = 0

    return depth_meters
```

---

### 2.3 步骤三：SAM3 模型推理

#### 2.3.1 SAM3 模型加载

```python
import torch
from sam3 import SAM3Model, SAM3Predictor

class SAM3Segmentor:
    def __init__(self, model_path, device='cuda'):
        """
        初始化SAM3分割器
        Args:
            model_path: SAM3模型权重路径
            device: 推理设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model = SAM3Model.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.predictor = SAM3Predictor(self.model)

    def segment_by_text(self, image, text_prompt):
        """
        根据文本提示进行分割
        Args:
            image: RGB图像 (H, W, 3)
            text_prompt: 类别文本提示，如 "red apple", "coffee mug"
        Returns:
            mask: 二值掩码 (H, W)
            confidence: 置信度分数
        """
        # 设置图像
        self.predictor.set_image(image)

        # 使用文本提示进行分割
        masks, scores, _ = self.predictor.predict(
            text_prompt=text_prompt,
            multimask_output=True
        )

        # 选择置信度最高的mask
        best_idx = scores.argmax()
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        return best_mask, best_score
```

---

### 2.4 步骤四：点云生成

#### 2.4.1 基于Mask提取目标深度

```python
def extract_masked_depth(depth_image, mask):
    """
    根据mask提取目标物体的深度值
    Args:
        depth_image: 深度图 (H, W)
        mask: 二值掩码 (H, W)
    Returns:
        masked_depth: 仅包含目标区域的深度图
    """
    masked_depth = np.zeros_like(depth_image)
    masked_depth[mask > 0] = depth_image[mask > 0]
    return masked_depth
```

#### 2.4.2 深度图转点云

```python
import open3d as o3d

def depth_to_pointcloud(depth_image, color_image, intrinsics, mask=None):
    """
    将深度图转换为点云
    Args:
        depth_image: 深度图 (H, W)，单位：米
        color_image: RGB图像 (H, W, 3)
        intrinsics: 相机内参 (fx, fy, cx, cy)
        mask: 可选的二值掩码
    Returns:
        pcd: Open3D点云对象
    """
    fx, fy, cx, cy = intrinsics
    height, width = depth_image.shape

    # 创建像素坐标网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 计算3D坐标
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 应用mask过滤
    if mask is not None:
        valid = (z > 0) & (mask > 0)
    else:
        valid = z > 0

    # 提取有效点
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = color_image[valid] / 255.0

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
```

---

### 2.5 步骤五：PCL点云滤波处理

#### 2.5.1 统计离群点滤波

```python
def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    统计离群点滤波
    Args:
        pcd: 输入点云
        nb_neighbors: 邻域点数
        std_ratio: 标准差倍数阈值
    """
    filtered_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return filtered_pcd
```

#### 2.5.2 半径离群点滤波

```python
def radius_outlier_removal(pcd, nb_points=16, radius=0.01):
    """
    半径离群点滤波
    Args:
        pcd: 输入点云
        nb_points: 半径内最少点数
        radius: 搜索半径 (米)
    """
    filtered_pcd, ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius
    )
    return filtered_pcd
```

#### 2.5.3 体素下采样

```python
def voxel_downsample(pcd, voxel_size=0.002):
    """
    体素下采样
    Args:
        pcd: 输入点云
        voxel_size: 体素大小 (米)
    """
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd
```

#### 2.5.4 完整滤波流程

```python
def filter_pointcloud(pcd):
    """完整的点云滤波流程"""
    # 1. 体素下采样
    pcd = voxel_downsample(pcd, voxel_size=0.002)

    # 2. 统计离群点滤波
    pcd = statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0)

    # 3. 半径离群点滤波
    pcd = radius_outlier_removal(pcd, nb_points=16, radius=0.01)

    return pcd
```

---

### 2.6 步骤六：二指夹爪抓取位姿估计

#### 2.6.1 点云主成分分析 (PCA)

```python
def compute_grasp_pose_pca(pcd):
    """
    基于PCA计算抓取位姿
    适用于规则形状物体
    """
    points = np.asarray(pcd.points)

    # 计算质心
    centroid = np.mean(points, axis=0)

    # PCA分析
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值排序（从大到小）
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 构建抓取坐标系
    approach = eigenvectors[:, 2]  # 最小特征值方向-接近方向
    binormal = eigenvectors[:, 0]  # 最大特征值方向-夹爪开合方向
    normal = np.cross(approach, binormal)

    # 构建旋转矩阵
    rotation_matrix = np.column_stack([binormal, normal, approach])

    return centroid, rotation_matrix
```

#### 2.6.2 计算抓取宽度

```python
def compute_grasp_width(pcd, grasp_direction):
    """
    计算沿抓取方向的物体宽度
    """
    points = np.asarray(pcd.points)
    projections = np.dot(points, grasp_direction)
    width = projections.max() - projections.min()
    return width + 0.01  # 添加安全余量
```

#### 2.6.3 坐标系变换

```python
def transform_to_base(grasp_pose_cam, T_cam_to_base):
    """
    将相机坐标系下的抓取位姿转换到机械臂基座坐标系
    """
    position_cam, rotation_cam = grasp_pose_cam

    # 构建4x4变换矩阵
    T_grasp_cam = np.eye(4)
    T_grasp_cam[:3, :3] = rotation_cam
    T_grasp_cam[:3, 3] = position_cam

    # 变换到基座坐标系
    T_grasp_base = T_cam_to_base @ T_grasp_cam

    position_base = T_grasp_base[:3, 3]
    rotation_base = T_grasp_base[:3, :3]

    return position_base, rotation_base
```

---

### 2.7 步骤七：机械臂执行抓取

#### 2.7.1 抓取执行流程

```python
class GraspExecutor:
    def __init__(self, robot_arm, gripper):
        self.robot = robot_arm
        self.gripper = gripper
        self.pre_grasp_offset = 0.10  # 预抓取位置偏移 (米)

    def execute_grasp(self, position, rotation, grasp_width):
        """
        执行抓取动作
        """
        # 1. 打开夹爪
        self.gripper.open()

        # 2. 移动到预抓取位置
        approach_dir = rotation[:, 2]
        pre_grasp_pos = position - approach_dir * self.pre_grasp_offset
        self.robot.move_to_pose(pre_grasp_pos, rotation)

        # 3. 直线运动到抓取位置
        self.robot.move_linear(position, rotation)

        # 4. 闭合夹爪
        self.gripper.close(width=grasp_width)

        # 5. 提升物体
        lift_pos = position.copy()
        lift_pos[2] += 0.15
        self.robot.move_linear(lift_pos, rotation)

        return True
```

---

## 3. 完整主程序流程

```python
class SAM3GraspSystem:
    def __init__(self, config):
        # 初始化各模块
        self.camera = RealSenseCamera()
        self.segmentor = SAM3Segmentor(config['sam3_model_path'])
        self.robot = RobotArm(config['robot_ip'])
        self.gripper = Gripper(config['gripper_port'])
        self.executor = GraspExecutor(self.robot, self.gripper)

        # 加载标定参数
        self.T_cam_to_base = np.load(config['calibration_path'])
        self.intrinsics = self.camera.get_intrinsics()

    def run(self, text_prompt):
        """
        执行一次完整的抓取流程
        """
        # Step 1: 获取图像
        color_image, depth_image = self.camera.get_frames()

        # Step 2: SAM3分割
        mask, confidence = self.segmentor.segment_by_text(
            color_image, text_prompt
        )
        if confidence < 0.5:
            print(f"分割置信度过低: {confidence}")
            return False

        # Step 3: 生成点云
        depth_meters = preprocess_depth(
            depth_image, self.camera.depth_scale
        )
        pcd = depth_to_pointcloud(
            depth_meters, color_image,
            self.intrinsics, mask
        )

        # Step 4: 点云滤波
        pcd = filter_pointcloud(pcd)

        # Step 5: 计算抓取位姿
        position_cam, rotation_cam = compute_grasp_pose_pca(pcd)
        grasp_width = compute_grasp_width(pcd, rotation_cam[:, 0])

        # Step 6: 坐标变换
        position_base, rotation_base = transform_to_base(
            (position_cam, rotation_cam), self.T_cam_to_base
        )

        # Step 7: 执行抓取
        success = self.executor.execute_grasp(
            position_base, rotation_base, grasp_width
        )

        return success
```

### 3.1 使用示例

```python
if __name__ == "__main__":
    config = {
        'sam3_model_path': './models/sam3_vit_h.pth',
        'robot_ip': '192.168.1.100',
        'gripper_port': '/dev/ttyUSB0',
        'calibration_path': './calibration/T_cam_to_base.npy'
    }

    system = SAM3GraspSystem(config)

    # 用户输入抓取目标
    text_prompt = input("请输入要抓取的物体: ")
    # 例如: "red apple", "coffee mug", "blue box"

    success = system.run(text_prompt)
    print(f"抓取{'成功' if success else '失败'}")
```

---

## 4. 关键注意事项

### 4.1 相机配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 分辨率 | 640x480 | 平衡精度与速度 |
| 帧率 | 30 fps | 实时性要求 |
| 深度范围 | 0.1-2.0m | 根据工作空间调整 |
| 对齐模式 | 深度对齐到彩色 | 确保像素对应 |

### 4.2 SAM3 推理优化

- 使用 GPU 加速推理（推荐 RTX 3060 及以上）
- 首次推理会较慢（模型加载），后续推理约 50-100ms
- 可使用 TensorRT 进一步优化推理速度

### 4.3 点云处理参数调优

| 滤波方法 | 参数 | 调优建议 |
|----------|------|----------|
| 统计滤波 | nb_neighbors=20, std_ratio=2.0 | 噪声多时减小std_ratio |
| 半径滤波 | nb_points=16, radius=0.01 | 点云稀疏时减小nb_points |
| 体素下采样 | voxel_size=0.002 | 精度要求高时减小体素 |

### 4.4 抓取安全措施

1. **碰撞检测**：执行前检查抓取路径是否与障碍物碰撞
2. **力控制**：夹爪闭合时监测夹持力，防止损坏物体
3. **工作空间限制**：设置机械臂运动范围，防止越界
4. **急停机制**：配置紧急停止按钮

---

## 5. 常见问题与解决方案

### 5.1 分割效果不佳

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 无法识别目标 | Prompt描述不准确 | 使用更具体的描述 |
| 分割边界模糊 | 光照条件差 | 改善环境光照 |
| 误分割 | 场景中有相似物体 | 添加颜色/位置限定词 |

### 5.2 点云质量问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 点云空洞 | 反光/透明材质 | 调整相机角度或贴标记 |
| 噪声过多 | 深度传感器精度 | 增强滤波参数 |
| 点云稀疏 | 距离过远 | 调整相机位置 |

### 5.3 抓取失败

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 位置偏差大 | 手眼标定不准 | 重新标定 |
| 抓取滑落 | 夹持力不足 | 调整夹爪力度 |
| 碰撞桌面 | 深度误差 | 添加安全高度偏移 |

---

## 6. 流程总结

```
┌────────────────────────────────────────────────────────────┐
│                    实时抓取流程                              │
├────────────────────────────────────────────────────────────┤
│  1. 用户输入 → 抓取类别 Prompt (如 "red apple")             │
│                         ↓                                  │
│  2. 相机采集 → RGB图像 + 深度图 (已对齐)                    │
│                         ↓                                  │
│  3. SAM3推理 → 目标物体 Mask                                │
│                         ↓                                  │
│  4. 点云生成 → Mask区域深度 → 3D点云                        │
│                         ↓                                  │
│  5. 点云滤波 → 去噪 + 下采样                                │
│                         ↓                                  │
│  6. 位姿估计 → PCA计算抓取位姿                              │
│                         ↓                                  │
│  7. 坐标变换 → 相机系 → 机械臂基座系                        │
│                         ↓                                  │
│  8. 执行抓取 → 预抓取 → 抓取 → 提升                         │
└────────────────────────────────────────────────────────────┘
```

---

**文档版本**: v1.0
**最后更新**: 2025-12-29
