import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from math import sqrt
from models.GCN import SKINNET
from dataset.format import parents

# 添加计算边索引偏移的函数
def calculate_edge_offsets(edge_index, batch_size, num_vertices, num_joints):
    num_edges = edge_index.shape[1]
    vertex_offsets = jt.arange(0, batch_size * num_vertices, dtype=jt.int32) * num_joints
    # 为每个顶点创建边索引
    # 首先创建一个包含所有顶点偏移的数组
    # 每行都是相同的偏移值
    offset_matrix = vertex_offsets.repeat(num_edges, 1)  
    offset_matrix = offset_matrix.transpose(1, 0) 
    flat_offsets = offset_matrix.reshape(-1) 
    src_offsets = flat_offsets.repeat(1) 
    dst_offsets = flat_offsets.repeat(1)  
    src_indices = edge_index[0].repeat(batch_size * num_vertices).astype(jt.int32)  
    dst_indices = edge_index[1].repeat(batch_size * num_vertices).astype(jt.int32)  
    src_indices = src_indices + src_offsets 
    dst_indices = dst_indices + dst_offsets  
    
    return jt.stack([src_indices, dst_indices], dim=0).astype(jt.int32)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)

class SkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.vertices_mlp = MLP(3, feat_dim)
        self.joints_mlp = MLP(3, feat_dim)
        
        self.final_mlp = MLP(2, feat_dim)

        self.skin_net = SKINNET(input_dim=feat_dim, aggr='add')
        
        # 创建骨架边索引
        self.create_skeleton_edges()
    
    def create_skeleton_edges(self):
        # 根据parents列表创建边索引
        edges = []
        for i, parent in enumerate(parents):
            if parent is not None:
                # 添加双向边
                edges.append([parent, i])  # 父节点到子节点
                edges.append([i, parent])  # 子节点到父节点
        
        # 转换为jittor张量，形状为(2, E)，确保使用int32类型
        self.edge_index = jt.array(edges, dtype=jt.int32).transpose()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # vertices (B, N, 3) joints (B, J, 3)
        
        B, N, _ = vertices.shape
        _, J, _ = joints.shape
        
        # 分别通过MLP映射vertices和joints
        vertices_feat = self.vertices_mlp(vertices)  # (B, N, feat_dim)
        joints_feat = self.joints_mlp(joints)  # (B, J, feat_dim)
        
        concat_feat = nn.softmax(vertices_feat @ joints_feat.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        
        # 计算距离
        vertices_expand = vertices.unsqueeze(2).repeat(1, 1, J, 1)  # (B, N, J, 3)
        joints_expand = joints.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, J, 3)
        distance = (vertices_expand - joints_expand).sum(dim=3, keepdims=True).sqrt()  # (B, N, J, 1)
        # 连接融合特征和距离特征
        fuse_feat = concat([concat_feat[..., None], distance], dim=3)  # (B, N, J, feat_dim*2)
        
        final_feat = self.final_mlp(fuse_feat)
        
        # 获取边索引的基本形状
        edge_index_with_offset = calculate_edge_offsets(self.edge_index, B, N, J)
 
        skin_cls_pred = self.skin_net(final_feat, edge_index_with_offset)
        skin_cls_pred = nn.softmax(skin_cls_pred.reshape(B, N, J),dim=-1)
        return skin_cls_pred



# Factory function to create models
def create_model(model_name='pct', feat_dim=256, **kwargs):
    if model_name == "pct":
        return SkinModel(feat_dim=feat_dim, num_joints=22)
    raise NotImplementedError()
