from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from typing import List, Union, Tuple
from collections import defaultdict
import os
import trimesh

from scipy.spatial.transform import Rotation as R
from .sampler import Sampler
from .exporter import Exporter

def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    """
    将轴角表示法转换为旋转矩阵

    参数:
        axis_angle: 轴角向量，表示旋转轴和旋转角度，例如[0,0,pi/2]表示绕z轴旋转90度

    返回:
        4x4的旋转矩阵
    """

    res = np.pad(R.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    assert res.ndim == 3
    res[:, -1, -1] = 1
    return res

def linear_blend_skinning(
    vertex: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=0,
    value: float=0.,
):
    '''
    线性混合蒙皮算法，用于骨骼动画顶点会先被转换到每个骨骼的局部空间
    然后应用每个骨骼的全局变换
    最后根据权重进行混合
    
    参数:
        vertex: (N, 3+pad) 顶点坐标
        matrix_local: (J, 4, 4) 局部变换矩阵
        matrix: (J, 4, 4) 全局变换矩阵
        skin: (N, J) 蒙皮权重，伪骨骼的值应为0
        pad: 填充维度
        value: 填充值
        
    返回:
        (N, 3) 变换后的顶点坐标
    '''
    J = matrix_local.shape[0]  # 获取骨骼数量
    # 对顶点坐标进行填充，通常用于齐次坐标
    padded = np.pad(vertex, ((0, 0), (0, pad)), 'constant', constant_values=(0, value))

    # 计算每个顶点相对于每个骨骼的局部坐标
    offset = (
        np.linalg.inv(matrix_local) @  # 求局部变换矩阵的逆
        np.tile(padded.transpose(), (J, 1, 1))  # 将顶点坐标复制J份
    )

    # 应用全局变换矩阵
    per_bone_matrix = matrix @ offset

    # 应用蒙皮权重
    weighted_per_bone_matrix = skin.T[:, np.newaxis, :] * per_bone_matrix

    # 对所有骨骼的影响进行求和
    g = np.sum(weighted_per_bone_matrix, axis=0)

    # 归一化并提取最终的3D坐标
    final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    return final.T

@dataclass
class Asset(Exporter):
    '''
    一个简单的资产类，用于加载网格、骨架和蒙皮
    '''
    # 数据类别
    cls: str
    # 数据ID
    id: int
    # 网格顶点，形状 (N, 3)，float32
    vertices: ndarray
    # 顶点法线，形状 (N, 3)，float32
    vertex_normals: Union[ndarray, None]
    # 网格面，形状 (F, 3)，面ID从0到F-1，int64
    faces: Union[ndarray, None]
    # 面法线，形状 (F, 3)，float32
    face_normals: Union[ndarray, None]
    # 骨骼关节，形状 (J, 3)，float32
    joints: Union[ndarray, None] = None
    # 关节蒙皮权重，形状 (N, J)，float32
    skin: Union[ndarray, None] = None
    # 关节的父级，None表示没有父级(根关节)
    # 确保parent[k] < k
    parents: Union[List[Union[int, None]], None] = None
    # 关节名称
    names: Union[List[str], None] = None
    # 骨骼的局部坐标
    matrix_local: Union[ndarray, None] = None

    down_sample_points: Union[ndarray, None] = None
    down_sample_normal: Union[ndarray, None] = None
    volumetric_geodesic_joint: Union[ndarray, None] = None
    volumetric_geodesic: Union[ndarray, None] = None
    
    def check_order(self) -> bool:
        """
        检查骨骼层次结构顺序是否正确
        确保每个关节的父关节索引小于自身索引
        """
        for i in range(self.J):
            if self.parents[i] is not None and self.parents[i] >= i:
                return False
        return True
    
    @property
    def N(self):
        '''
        顶点数量
        '''
        return self.vertices.shape[0]
    
    @property
    def F(self):
        '''
        面数量
        '''
        return self.faces.shape[0]
    
    @property
    def J(self):
        '''
        关节数量
        '''
        return self.joints.shape[0]
    
    @staticmethod
    def load(path: str) -> 'Asset':
        """
        从文件加载资产
        
        参数:
            path: 文件路径
            
        返回:
            Asset对象
        """
        data = np.load(path, allow_pickle=True)
        d = {n: v[()] for (n, v) in data.items()}
        return Asset(**d)
    
    def set_order_by_names(self, new_names: List[str]):
        """
        根据新的名称列表重新排序骨骼
        检查新的名称列表长度是否与原列表相同
        创建新旧顺序的映射关系
        重新排列以下数据：
        
        参数:
            new_names: 新的关节名称列表
        """
        assert len(new_names) == len(self.names)
        name_to_id = {name: id for (id, name) in enumerate(self.names)}
        new_name_to_id = {name: id for (id, name) in enumerate(new_names)}
        perm = []
        new_parents = []
        for (new_id, name) in enumerate(new_names):
            perm.append(name_to_id[name])
            pid = self.parents[name_to_id[name]]
            if new_id == 0:
                assert pid is None, 'first bone is not root bone'
            else:
                pname = self.names[pid]
                pid = new_name_to_id[pname]
                assert pid < new_id, 'new order does not form a tree'
            new_parents.append(pid)
        
        if self.joints is not None:
            self.joints = self.joints[perm]
        self.parents = new_parents
        if self.skin is not None:
            self.skin = self.skin[:, perm]
        if self.matrix_local is not None:
            self.matrix_local = self.matrix_local[perm]
        self.names = new_names
    
    def get_random_matrix_basis(self, random_pose_angle: float) -> ndarray:
        '''
        生成随机姿势的基础变换矩阵
        
        参数:
            random_pose_angle: 随机角度范围（度）
            
        返回:
            随机姿势矩阵
        '''
        matrix_basis = axis_angle_to_matrix((np.random.rand(self.J, 3) - 0.5) * random_pose_angle / 180 * np.pi * 2).astype(np.float32)
        return matrix_basis
        
    
    def apply_matrix_basis(self, matrix_basis: ndarray):
        '''
        应用姿势变换到骨架
        
        参数:
            matrix_basis: (J, 4, 4) 基础变换矩阵
        '''
        matrix_local = self.matrix_local
        if matrix_local is None:
            # 如果没有局部矩阵，创建单位矩阵并设置平移部分
            matrix_local = np.zeros((self.J, 4, 4))
            matrix_local[:, 0, 0] = 1.
            matrix_local[:, 1, 1] = 1.
            matrix_local[:, 2, 2] = 1.
            matrix_local[:, 3, 3] = 1.
            for i in range(self.J):
                matrix_local[i, :3, 3] = self.joints[i]
        
        # 计算全局变换矩阵
        matrix = np.zeros((self.J, 4, 4))
        for i in range(self.J):
            if i==0:
                # 根关节直接应用局部变换和基础变换
                matrix[i] = matrix_local[i] @ matrix_basis[i]
            else:
                # 非根关节需要考虑父关节的变换
                pid = self.parents[i]
                matrix_parent = matrix[pid]
                matrix_local_parent = matrix_local[pid]
                
                matrix[i] = (
                    matrix_parent @
                    (np.linalg.inv(matrix_local_parent) @ matrix_local[i]) @
                    matrix_basis[i]
                )
        self.joints = matrix[:, :3, 3]
        # 应用线性混合蒙皮算法变形顶点
        vertices = linear_blend_skinning(self.vertices, matrix_local, matrix, self.skin, pad=1, value=1.)
        # 更新局部矩阵
        self.matrix_local = matrix
        
        # 使用trimesh重新计算法线
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        self.vertices = vertices
        self.vertex_normals = mesh.vertex_normals.copy()
        self.face_normals = mesh.face_normals.copy()
    
    def sample(self, sampler: Sampler) -> 'SampledAsset':
        '''
        对资产进行采样，用于模型输入
        
        参数:
            sampler: 采样器对象
            
        返回:
            SampledAsset对象
        '''
        vertex_groups = {}
        if self.skin is not None:
            vertex_groups['skin'] = self.skin.copy()
        sampled_vertices, sampled_normal, vertex_groups = sampler.sample(
            vertices=self.vertices,
            vertex_normals=self.vertex_normals,
            face_normals=self.face_normals,
            vertex_groups=vertex_groups,
            faces=self.faces,
        )
        
        # 法线归一化
        eps = 1e-6
        sampled_normal = sampled_normal / (np.linalg.norm(sampled_normal, axis=1, keepdims=True) + eps)
        sampled_normal = np.nan_to_num(sampled_normal, nan=0., posinf=0., neginf=0.)
        
        return SampledAsset(
            cls=self.cls,
            id=self.id,
            vertices=sampled_vertices,
            normals=sampled_normal,
            joints=self.joints,
            skin=vertex_groups.get('skin', None),
            parents=self.parents,
            names=self.names,
        )

    def sample_geo(self, sampler: Sampler) -> 'SampledAsset':
        '''
        对资产进行采样，用于模型输入
        
        参数:
            sampler: 采样器对象
            
        返回:
            SampledAsset对象
        '''
        vertex_groups = {}
        points_groups = {}
        if self.skin is not None:
            vertex_groups['skin'] = self.skin.copy()
        if self.volumetric_geodesic_joint is not None:
            points_groups['volumetric_geodesic_joint'] = self.volumetric_geodesic_joint.copy()
        sampled_vertices, sampled_normal, vertex_groups,sampled_points, sampled_normals, points_groups = sampler.sample(
            vertices= self.vertices,
            vertex_normals= self.vertex_normals,
            face_normals= self.face_normals,
            vertex_groups= vertex_groups,
            faces= self.faces,
            points= self.down_sample_points,      
            points_normals= self.down_sample_normal, 
            points_groups= points_groups, 
        )
        
        # 法线归一化
        eps = 1e-6
        sampled_normal = sampled_normal / (np.linalg.norm(sampled_normal, axis=1, keepdims=True) + eps)
        sampled_normal = np.nan_to_num(sampled_normal, nan=0., posinf=0., neginf=0.)
        
        return SampledAsset(
            cls=self.cls,
            id=self.id,
            vertices=sampled_vertices,
            normals=sampled_normal,
            down_sample_points=sampled_points,
            down_sample_normal=sampled_normals,
            joints=self.joints,
            skin=vertex_groups.get('skin', None),
            volumetric_geodesic=points_groups.get('volumetric_geodesic_joint', None),
            parents=self.parents,
            names=self.names,
        )
    
    def export_pc(self, path: str, with_normal: bool=True, size=0.01):
        '''
        导出点云
        
        参数:
            path: 导出路径
            with_normal: 是否包含法线
            size: 点大小
        '''
        if with_normal:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=self.vertex_normals, size=size)
        else:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=None, size=size)
    
    def export_mesh(self, path: str):
        '''
        导出网格
        
        参数:
            path: 导出路径
        '''
        self._export_mesh(vertices=self.vertices, faces=self.faces, path=path)
    
    def export_skeleton(self, path: str):
        '''
        导出骨架
        
        参数:
            path: 导出路径
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect: bool=True,
        extrude_from_parent: bool=True,
    ):
        '''
        导出带蒙皮的完整模型为FBX格式
        
        参数:
            path: 导出路径
            extrude_size: 挤出大小
            group_per_vertex: 每个顶点的组数，-1表示不限制
            add_root: 是否添加根骨骼
            do_not_normalize: 是否不进行归一化
            try_connect: 是否尝试连接骨骼
            extrude_from_parent: 是否从父骨骼挤出
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            faces=self.faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect,
            extrude_from_parent=extrude_from_parent,
        )
    
    def export_animation(
        self,
        path: str,
        matrix_basis: ndarray,
        offset: ndarray,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect=True,
    ):
        '''
        导出动画
        
        参数:
            path: 导出路径
            matrix_basis: 基础变换矩阵
            offset: 偏移量
            extrude_size: 挤出大小
            group_per_vertex: 每个顶点的组数，-1表示不限制
            add_root: 是否添加根骨骼
            do_not_normalize: 是否不进行归一化
            try_connect: 是否尝试连接骨骼
        '''
        self._export_animation(
            path=path,
            matrix_basis=matrix_basis,
            offset=offset,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            faces=self.faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect,
        )

@dataclass
class SampledAsset(Exporter):
    '''
    采样后的资产类，用于模型输入
    '''
    
    # 数据类别
    cls: str
    
    # 数据ID
    id: int
    
    # 采样后的顶点，形状 (N, 3)，float32
    vertices: ndarray
    
    # 采样后的法线，形状 (N, 3)，float32
    normals: ndarray
    
    # 骨骼关节，形状 (J, 3)，float32
    joints: Union[ndarray, None] = None
    
    # 关节蒙皮权重，形状 (N, J)，float32
    skin: Union[ndarray, None] = None
    
    # 关节的父级，None表示没有父级(根关节)
    # 确保parent[k] < k
    parents: Union[List[Union[int, None]], None] = None
    
    # 关节名称
    names: Union[List[str], None] = None

    down_sample_points: Union[ndarray, None] = None
    down_sample_normal: Union[ndarray, None] = None
    volumetric_geodesic_joint: Union[ndarray, None] = None
    volumetric_geodesic: Union[ndarray, None] = None
    
    @property
    def N(self):
        '''
        顶点数量
        '''
        return self.vertices.shape[0]
    
    @property
    def J(self):
        '''
        关节数量
        '''
        return self.joints.shape[0]
    
    def export_pc(self, path: str, with_normal: bool=True, size=0.01):
        '''
        导出点云
        
        参数:
            path: 导出路径
            with_normal: 是否包含法线
            size: 点大小
        '''
        if with_normal:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=self.normals, size=size)
        else:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=None, size=size)
    
    def export_skeleton(self, path: str):
        '''
        导出骨架
        
        参数:
            path: 导出路径
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect: bool=True
    ):
        '''
        导出带蒙皮的点云为FBX格式
        
        参数:
            path: 导出路径
            extrude_size: 挤出大小
            group_per_vertex: 每个顶点的组数，-1表示不限制
            add_root: 是否添加根骨骼
            do_not_normalize: 是否不进行归一化
            try_connect: 是否尝试连接骨骼
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            faces=None,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect
        )