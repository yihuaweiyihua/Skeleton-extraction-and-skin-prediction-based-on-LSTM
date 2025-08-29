import jittor as jt
import numpy as np
import os
from jittor.dataset import Dataset

import os
from typing import List, Dict, Callable, Union

from .asset import Asset
from .sampler import Sampler
from jittor import nn

def transform(asset: Asset):
    """
    函数同时处理顶点（vertices）和关节点（joints）数据
    使用相同的变换参数（center和scale）确保顶点和关节的相对位置关系保持不变
    最后更新了matrix_local矩阵的平移部分（第4列的前3个元素），这是因为它存储了关节的全局变换信息
    """
    # Find min and max values for each dimension of points
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Calculate the scale factor to normalize to [-1, 1]
    # We take the maximum range across all dimensions to preserve aspect ratio
    scale = np.max(max_vals - min_vals) / 2
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (asset.vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        normalized_joints = (asset.joints - center) / scale
    else:
        normalized_joints = None
    
    asset.vertices  = normalized_vertices
    asset.joints    = normalized_joints
    # remember to change matrix_local !
    #asset.matrix_local[:, :3, 3] = normalized_joints

class RigDataset(Dataset):
    '''
    一个用于处理人体骨骼数据的简单数据集类。
    用于加载和处理包含顶点、法线、关节和蒙皮权重的3D模型数据。
    '''
    def __init__(
        self,
        data_root: str,          # 数据根目录
        paths: List[str],        # 数据文件路径列表
        train: bool,             # 是否为训练集
        batch_size: int,         # 批次大小
        shuffle: bool,           # 是否打乱数据
        sampler: Sampler,        # 采样器，用于对点云进行采样
        transform: Union[Callable, None] = None,  # 可选的变换函数
        return_origin_vertices: bool = False ,
        is_geo: bool = False,
        random_pose: bool = False,
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # 避免与Dataset的sampler属性冲突
        self.transform  = transform
        self.is_geo   = is_geo
        self.random_pose = random_pose
        self.return_origin_vertices = return_origin_vertices
        
        # 预加载所有矩阵数据
        if self.random_pose:
            self.all_matrix_basis = []
            for file_idx in range(10):  # 加载0-9的npz文件
                npz_data = np.load(f'data/track/{file_idx}.npz')
                matrix_basis_data = npz_data['matrix_basis']  # shape: (frame, J, 4, 4)
                self.all_matrix_basis.append(matrix_basis_data)
            # 将所有矩阵合并为一个大数组，便于随机选择
            self.all_matrix_basis = np.concatenate(self.all_matrix_basis, axis=0)
        # 设置数据集的基本属性
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def __getitem__(self, index) -> Dict:
        """
        获取数据集中的一个样本
        
        参数:
            index (int): 样本的索引
            
        返回:
            data (Dict): 包含以下键的字典:
                - vertices: jt.Var, 形状为(B, N, 3)的点云数据
                - normals: jt.Var, 形状为(B, N, 3)的点云法线
                - joints: jt.Var, 形状为(B, J, 3)的关节位置
                - skin: jt.Var, 形状为(B, J, J)的蒙皮权重
        """
        
        # 加载数据文件
        path = self.paths[index]
        asset = Asset.load(os.path.join(self.data_root, path))
        if self.random_pose and np.random.rand() < 0.6:
            # 从预加载的矩阵中随机选择一个
            random_idx = np.random.randint(0, len(self.all_matrix_basis))
            matrix_basis = self.all_matrix_basis[random_idx]
            asset.apply_matrix_basis(matrix_basis)
        # 应用变换（如果存在）
        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy()).float32()
        sampled_asset = asset
        # 使用采样器对资产进行采样
        if self.is_geo:
            sampled_asset = asset.sample_geo(sampler=self._sampler)
            down_sample_points = jt.array(sampled_asset.down_sample_points).float32()
            down_sample_normal = jt.array(sampled_asset.down_sample_normal).float32()
            volumetric_geodesic = jt.array(sampled_asset.volumetric_geodesic).float32()
        else:
            sampled_asset = asset.sample(sampler=self._sampler)
            # 转换数据为jittor张量
        vertices    = jt.array(sampled_asset.vertices).float32()
        normals     = jt.array(sampled_asset.normals).float32()

        # 处理关节数据（如果存在）
        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        # 处理蒙皮权重数据（如果存在）
        if sampled_asset.skin is not None:
            skin = jt.array(sampled_asset.skin).float32()
            #skin = nn.softmax(skin, dim=-1)
        else:
            skin   = None

        # points = jt.contrib.concat([joints,vertices], dim=0)
        # adj = get_matirx(points)
        # label = label_matirx2(adj)
        # adj = update_matirx(adj)

        # 构建返回的字典
        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
            # 'adj': adj,
            # 'label': label
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None :
            res['skin'] = skin
        if self.return_origin_vertices :
            res['origin_vertices'] = origin_vertices
        if  self.is_geo:
            res['volumetric_geodesic'] = volumetric_geodesic
            res['down_sample_points'] = down_sample_points
            res['down_sample_normal'] = down_sample_normal

        return res
    
    def collate_batch(self, batch):

        # 如果需要返回原始顶点数据，进行填充处理
        if self.return_origin_vertices:
            # 找到最大顶点数量
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            # 对每个样本进行填充
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)

# Example usage of the dataset
def get_dataloader(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    is_geo: bool = False,
    random_pose: bool = False,
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        
    Returns:
        dataset (RigDataset): The dataset
    """
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        is_geo = is_geo,
        random_pose= random_pose,
    )
    
    return dataset
