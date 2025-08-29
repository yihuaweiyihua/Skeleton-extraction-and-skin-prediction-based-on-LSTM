import numpy as np
from numpy import ndarray
from typing import Tuple
from abc import ABC, abstractmethod

class Sampler(ABC):
    """采样器基类，定义了采样器的基本接口"""
    def __init__(self):
        pass
    
    def _sample_barycentric(
        self,
        vertex_groups: ndarray,  # 顶点组数据
        faces: ndarray,          # 面片索引
        face_index: ndarray,     # 选中的面片索引
        random_lengths: ndarray,  # 随机生成的重心坐标权重
    ):
        """
        使用重心坐标（barycentric coordinates）
        对三角形面片上的顶点属性进行插值计算。
        
        参数:
            vertex_groups: 顶点属性数据
            faces: 面片索引数组
            face_index: 选中的面片索引
            random_lengths: 随机生成的重心坐标权重
            
        返回:
            插值计算后的属性值
        """
        # 获取面片第一个顶点的属性值作为原点
        v_origins = vertex_groups[faces[face_index, 0]]
        # 获取面片其他两个顶点的属性值
        v_vectors = vertex_groups[faces[face_index, 1:]]
        # 计算相对于原点的向量
        v_vectors -= v_origins[:, np.newaxis, :]
        
        # 使用重心坐标进行插值
        sample_vector = (v_vectors * random_lengths).sum(axis=1)
        v_samples = sample_vector + v_origins
        return v_samples
    
    @abstractmethod
    def sample(
        self,
        vertices: ndarray,      # 顶点坐标数组 (N, 3)
        vertex_normals: ndarray, # 顶点法向量数组 (N, 3)
        face_normals: ndarray,   # 面法向量数组 (F, 3)
        vertex_groups: dict[str, ndarray], # 顶点组字典 {组名: 顶点属性数组}
        faces: ndarray,          # 面片索引数组 (F, 3)
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        """
        采样接口，需要被子类实现
        
        参数:
            vertices: 顶点坐标数组，形状为(N, 3)
            vertex_normals: 顶点法向量数组，形状为(N, 3)
            face_normals: 面法向量数组，形状为(F, 3)
            vertex_groups: 顶点组字典，每个值形状为(N, x)
            faces: 面片索引数组，形状为(F, 3)
            
        返回:
            vertices: 采样后的顶点坐标
            vertex_normals: 采样后的法向量
            vertex_groups: 采样后的顶点组属性
        """
        return vertices, vertex_normals, vertex_groups

class SamplerRamdon(Sampler):
    def __init__(self, num_samples: int, vertex_samples: int, points_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.vertex_samples = vertex_samples
        self.points_samples = points_samples
    
    def sample(
        self,
        vertices: ndarray,      # 顶点坐标数组 (N, 3)
        vertex_normals: ndarray, # 顶点法向量数组 (N, 3)
        face_normals: ndarray,   # 面法向量数组 (F, 3)
        vertex_groups: dict[str, ndarray], # 顶点组字典 {组名: 顶点属性数组}
        faces: ndarray,          # 面片索引数组 (F, 3)
        points: ndarray,      
        points_normals: ndarray, 
        points_groups: dict[str, ndarray], 
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        if self.num_samples==-1:
            return vertices, vertex_normals, vertex_groups
        
        # 第一步：从原始顶点中随机采样
        num_samples = self.num_samples
        perm = np.random.permutation(vertices.shape[0])  # 生成随机排列
        vertex_samples = min(self.vertex_samples, vertices.shape[0])  # 确保不超过顶点总数
        num_samples -= vertex_samples  # 更新剩余需要采样的数量
        perm = perm[:vertex_samples]  # 选择前vertex_samples个随机索引
        n_vertices = vertices[perm]  # 获取选中的顶点坐标
        n_normal = vertex_normals[perm]  # 获取选中的顶点法向量
        n_v = {name: v[perm] for name, v in vertex_groups.items()}  # 获取选中的顶点组属性
        
        # 第二步：从表面随机采样
        perm = np.random.permutation(num_samples)
        # 调用surface采样函数获取表面采样点
        vertex_samples, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=vertices,
            faces=faces,
            return_weight=True,
        )
        # 合并两种采样结果
        vertex_samples = np.concatenate([n_vertices, vertex_samples], axis=0)
        normal_samples = np.concatenate([n_normal, face_normals[face_index]], axis=0)
        # 处理顶点组属性
        vertex_groups_samples = {}
        for n, v in vertex_groups.items():
            # 使用重心坐标插值计算采样点的属性值
            g = self._sample_barycentric(
                vertex_groups=v,
                faces=faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )
            vertex_groups_samples[n] = np.concatenate([n_v[n], g], axis=0)

        # 如果num_samples为-1，返回原始数据
        if self.points_samples == -1:
            return points, points_normals, points_groups
        
        # 从原始顶点中随机采样
        num_samples = min(self.points_samples, points.shape[0])  # 确保不超过顶点总数
        perm = np.random.permutation(points.shape[0])[:num_samples]  # 生成随机排列并选择指定数量的索引
        
        # 获取采样的顶点数据
        sampled_points = points[perm]  # 获取选中的顶点坐标
        sampled_normals = points_normals[perm]  # 获取选中的顶点法向量
        sampled_groups = {name: v[perm] for name, v in points_groups.items()}  # 获取选中的顶点组属性
        
        return vertex_samples, normal_samples, vertex_groups_samples,sampled_points, sampled_normals, sampled_groups

class SamplerMix(Sampler):
    '''
    混合采样器：结合了两种采样策略
    1. 首先从原始顶点中随机选择vertex_samples个样本
    2. 然后从表面随机采样(num_vertices-vertex_samples)个点
    这种混合策略可以保证既保留原始顶点信息，又能获得均匀的表面采样点
    '''
    def __init__(self, num_samples: int, vertex_samples: int):
        """
        初始化混合采样器
        
        参数:
            num_samples: 总采样点数量
            vertex_samples: 从原始顶点中采样的数量
        """
        super().__init__()
        self.num_samples = num_samples
        self.vertex_samples = vertex_samples
    
    def sample(
        self,
        vertices: ndarray,      # 顶点坐标数组 (N, 3)
        vertex_normals: ndarray, # 顶点法向量数组 (N, 3)
        face_normals: ndarray,   # 面法向量数组 (F, 3)
        vertex_groups: dict[str, ndarray], # 顶点组字典 {组名: 顶点属性数组}
        faces: ndarray,          # 面片索引数组 (F, 3)
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        """
        执行混合采样策略
        
        参数:
            vertices: 顶点坐标数组，形状为(N, 3)
            vertex_normals: 顶点法向量数组，形状为(N, 3)
            face_normals: 面法向量数组，形状为(F, 3)
            vertex_groups: 顶点组字典，每个值形状为(N, x)
            faces: 面片索引数组，形状为(F, 3)
            
        返回:
            vertices: 采样后的顶点坐标
            vertex_normals: 采样后的法向量
            vertex_groups: 采样后的顶点组属性
        """
        # 如果num_samples为-1，返回原始数据
        if self.num_samples==-1:
            return vertices, vertex_normals, vertex_groups
        
        # 第一步：从原始顶点中随机采样
        num_samples = self.num_samples
        perm = np.random.permutation(vertices.shape[0])  # 生成随机排列
        vertex_samples = min(self.vertex_samples, vertices.shape[0])  # 确保不超过顶点总数
        num_samples -= vertex_samples  # 更新剩余需要采样的数量
        perm = perm[:vertex_samples]  # 选择前vertex_samples个随机索引
        n_vertices = vertices[perm]  # 获取选中的顶点坐标
        n_normal = vertex_normals[perm]  # 获取选中的顶点法向量
        n_v = {name: v[perm] for name, v in vertex_groups.items()}  # 获取选中的顶点组属性
        
        # 第二步：从表面随机采样
        perm = np.random.permutation(num_samples)
        # 调用surface采样函数获取表面采样点
        vertex_samples, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=vertices,
            faces=faces,
            return_weight=True,
        )
        # 合并两种采样结果
        vertex_samples = np.concatenate([n_vertices, vertex_samples], axis=0)
        normal_samples = np.concatenate([n_normal, face_normals[face_index]], axis=0)
        # 处理顶点组属性
        vertex_groups_samples = {}
        for n, v in vertex_groups.items():
            # 使用重心坐标插值计算采样点的属性值
            g = self._sample_barycentric(
                vertex_groups=v,
                faces=faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )
            vertex_groups_samples[n] = np.concatenate([n_v[n], g], axis=0)
        return vertex_samples, normal_samples, vertex_groups_samples

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    return_weight: bool=False,
):
    '''
    根据面片面积进行加权随机采样
    
    参数:
        num_samples: 需要采样的点数量
        vertices: 顶点坐标数组
        faces: 面片索引数组
        return_weight: 是否返回采样权重
        
    返回:
        vertex_samples: 采样得到的点坐标
        face_index: 采样点所在的面片索引（如果return_weight=True）
        random_lengths: 采样点的重心坐标（如果return_weight=True）
    '''
    # 计算每个面片的面积作为采样权重
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    face_weight = (face_weight * face_weight).sum(axis=1)
    
    # 根据面片面积进行加权随机采样
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # 将三角形转换为原点+两个向量的形式
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # 获取选中面片的原点和向量
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    
    # 生成随机重心坐标
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    # 确保重心坐标在有效范围内
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    # 使用重心坐标计算采样点位置
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    
    if not return_weight:
        return vertex_samples
    return vertex_samples, face_index, random_lengths