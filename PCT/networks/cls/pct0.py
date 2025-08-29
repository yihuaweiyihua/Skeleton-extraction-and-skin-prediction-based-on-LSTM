import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points
from PCT.networks.lstm.lstm import SkeletonTreeLSTM

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = jt.init.eye(d)[None, :, :]
    loss = jt.bmm(trans, trans.transpose(2, 1) - I).pow(2).sum()/trans.size()[0]
    return loss

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    # xyz = xyz.contiguous()
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    # print ('fps size=', fps_idx.size())
    # fps_idx = sampler(xyz).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

def sample_and_group_normal(npoint, nsample, xyz, points, normal,no_down=False):
    B, N, _ = xyz.shape

    if not no_down:
        fps_sampler = FurthestPointSampler(npoint)
        _, fps_idx = fps_sampler(xyz)  
        new_xyz = index_points(xyz, fps_idx)      
        new_points_center = index_points(points, fps_idx) 
        new_normal = index_points(normal, fps_idx) 
    else:
        new_xyz = xyz
        new_points_center = points
        new_normal = normal
        fps_idx = None

    K = min(4 * nsample, N)
    idx = knn_point(K, xyz, new_xyz)           
    grouped_normal = index_points(normal, idx) 

    norm_center = jt.normalize(new_normal, dim=-1)          
    norm_group  = jt.normalize(grouped_normal, dim=-1)     
    cosine_sim = jt.sum(norm_group * norm_center.unsqueeze(2), dim=-1)  

    valid_mask = cosine_sim > 0
    sim_masked = jt.where(valid_mask, cosine_sim, -1)  
    _, sorted_idx = jt.topk(sim_masked, nsample, dim=-1)

    sel_xyz = index_points(xyz, sorted_idx)        
    sel_points = index_points(points, sorted_idx)   
    points_norm = sel_points - new_points_center.view(B, npoint, 1, -1)
    new_points = concat([points_norm, new_points_center.view(B, npoint, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)

    return new_xyz, new_points, new_normal,(fps_idx, sorted_idx)


class Point_Transformer_lstm(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_lstm, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = SkeletonTreeLSTM(input_size=256, hidden_size=256, num_joints=52)

        self.relu = nn.ReLU()
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(scale=0.2))
                                   
        self.pt_last = Point_Transformer_Last()

        self.convs1 = nn.Conv1d(256 * 3, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256,52, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

    def execute(self, x, normal):
        # x [B, 3, N] normal [B, 3, N]
        xyz = x.permute(0, 2, 1)
        
        x0 = concat([x, normal], dim=1) # B, 6, N
        x = self.relu(self.bn1(self.conv1(x0))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)      
        normal = normal.permute(0, 2, 1)
        batch_size, _, _ = xyz.size()

        new_xyz, new_feature, new_normal,(fps_idx0, sorted_idx0) = sample_and_group_normal(npoint=4096, nsample=32, xyz=xyz, points=x, normal=normal,no_down=True) 
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature, new_normal,(fps_idx1, sorted_idx1) = sample_and_group_normal(npoint=4096, nsample=32, xyz=new_xyz, points=feature, normal=new_normal,no_down=True) 
        feature_1 = self.gather_local_1(new_feature)
        # add position embedding on each layer
        x = self.pt_last(feature_1, new_xyz)
        x = concat([x, feature_1], dim=1)
        x = self.conv_fuse(x)

        x_max = jt.max(x, 2)
        x_avg = jt.mean(x, 2)
        _,_,N = x.size()
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = concat((x_max_feature, x_avg_feature), 1) # 1024 + 64
        x_seg = concat((x, x_global_feature), 1)
        x_seg = self.relu(self.bns1(self.convs1(x_seg)))
        x_seg = self.dp1(x_seg)
        x_seg = self.relu(self.bns2(self.convs2(x_seg)))
        skeleton_score = self.convs3(x_seg)
        skeleton_score = skeleton_score.permute(0, 2, 1)
        row_joints = self.lstm(x, new_xyz, skeleton_score)
        
        return row_joints, skeleton_score, new_xyz, (fps_idx0, fps_idx1)

class Point_Transformer(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # Store original input for xyz coordinates
        x_input = x
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x



class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
    def execute(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        #xyz = self.conv_pos(xyz)
        # end
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        
        x = concat((x1, x2, x3, x4), dim=1)

        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
      # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # Add a projection for xyz coordinates
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # Project xyz to the same channel dimension as x
        xyz_feat = self.xyz_proj(xyz)
        
        # Now we can safely add them
        x = x + xyz_feat
        
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = nn.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = nn.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 


    network = Point_Transformer()
    out_logits = network(input_points)
    print (out_logits.shape)

