import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

from jittor_geometric.nn.conv.message_passing import MessagePassing
from jittor_geometric.utils.loop import add_self_loops

def MLP(channels, batch_norm=True):
    if batch_norm:
        return nn.Sequential(*[nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i], momentum=0.1))
                            for i in range(1, len(channels))])
    else:
        return nn.Sequential(*[nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU()) for i in range(1, len(channels))])

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='max'):
        super(EdgeConv, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn

    def execute(self, x, edge_index):
        """"""
        x = x.reshape(-1, self.in_channels) if x.ndim > 2 else x
        #edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(jt.concat([x_i, (x_j - x_i)], dim=1))

    def update(self, aggr_out):
        aggr_out = aggr_out.reshape(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class GCU(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(GCU, self).__init__()
        self.edge_conv_tpl = EdgeConv(in_channels=in_channels, out_channels=out_channels,
                                      nn=MLP([in_channels * 2, out_channels]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def execute(self, x, tpl_edge_index):  
        x_tpl = self.edge_conv_tpl(x, tpl_edge_index)
        x_out = self.mlp(x_tpl)
        return x_out
    
class SKINNET(nn.Module):
    def __init__(self, input_dim=3, aggr='mean'):
        super(SKINNET, self).__init__()
        self.multi_layer_tranform1 = MLP([input_dim, 128, 64])
        self.gcu1 = GCU(in_channels=64, out_channels=512, aggr=aggr)
        self.gcu2 = GCU(in_channels=512, out_channels=256, aggr=aggr)
        self.gcu3 = GCU(in_channels=256, out_channels=256, aggr=aggr)
        self.multi_layer_tranform2 = MLP([512, 512, 1024])

        self.cls_branch = nn.Sequential(
            nn.Linear(1024 + 256, 1024), 
            nn.ReLU(), 
            nn.BatchNorm1d(1024), 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )

    def execute(self, vertex_features, tpl_edge_index):
        B, NJ, feat_dim = vertex_features.shape
        raw_input = vertex_features.reshape(B*NJ, feat_dim)
        x_0 = self.multi_layer_tranform1(raw_input) # (N, 64)
        x_1 = self.gcu1(x_0, tpl_edge_index) # （N, 512）
        x_global = self.multi_layer_tranform2(x_1) # (N, 1024)
        x_dim = x_global.shape[1] 
        x_global_reshaped = x_global.reshape(B, NJ, x_dim)

        # 计算平均值
        x_global_mean_per_batch = jt.mean(x_global_reshaped, dim=1, keepdims=True)  # (B, 1, features)
        x_global_mean_expanded = x_global_mean_per_batch.expand(-1, NJ, -1)  # (B, N*J, features)

        # 重塑回原始形状
        x_global_mean = x_global_mean_expanded.reshape(B*NJ,-1)

        x_2 = self.gcu2(x_1, tpl_edge_index)
        x_3 = self.gcu3(x_2, tpl_edge_index)
        
        # 连接并重塑以适应分类分支
        x_concat = jt.concat([x_3, x_global_mean], dim=1)  # (N, 256+1024)
        x_4 = x_concat  # 已经是正确的形状 (N, 256+1024)

        skin_cls_pred = self.cls_branch(x_4)
        
        return skin_cls_pred


if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    jt.set_seed(42)
    
    # 设置参数
    batch_size = 2  # 批次大小
    num_points = 100  # 每个批次的点数
    feature_dim = 3  # 特征维度
    
    # 创建随机输入数据
    pos = jt.array(np.random.randn(num_points, feature_dim).astype(np.float32)) # (N, 3)
    
    # 创建随机边索引 (2, E)
    # 为简单起见，我们创建一些随机连接
    num_edges = 100
    tpl_edge_index = jt.array(np.random.randint(0, num_points, (2, num_edges)), dtype=jt.int32) 
    geo_edge_index = jt.array(np.random.randint(0, num_points, (2, num_edges)), dtype=jt.int32)
    
    # 创建随机标签
    target = jt.array(np.random.randint(0, 2, (num_points, 1)).astype(np.float32))
    
    # 创建模型和优化器
    model = SKINNET(aggr='add')
    optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
    
    # 前向传播
    output = model(pos, tpl_edge_index, geo_edge_index)
    
    # 计算损失
    loss = nn.binary_cross_entropy_with_logits(output, target)
    
    # 反向传播
    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()
    
    # 打印结果
    print("\n测试结果:")
    print(f"输出形状: {output.shape}")
    print(f"损失值: {loss.item()}")
    
    # 检查梯度
    print("\n梯度检查:")