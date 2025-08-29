import jittor as jt
import jittor.nn as nn

class SkeletonTreeLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_joints=22):
        super(SkeletonTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.layer = 2
        
        self.lstm = nn.LSTM(256, 256, self.layer, batch_first=True)

    def execute(self, x, origin_x, skeleton_score):
        # x: [batch_size, 256, N]
        batch_size, N = x.shape[0], x.shape[2]
        origin_x = origin_x.permute(0,2,1)
        
        skeleton_score = nn.softmax(skeleton_score, dim=2)
        _, idx = jt.topk(skeleton_score.permute(0,2,1), 500, dim=2)
        x0 = jt.max(x, 2)
        # 存储每个节点的3D位置
        all_joints = jt.zeros((batch_size, self.num_joints, 3))
        
        states = (jt.zeros(self.layer, origin_x.shape[0], 256),
                  jt.zeros(self.layer, origin_x.shape[0], 256))
        for i in range(52):
            idx_ = idx[:,i,:]
            input = jt.gather(x, 2, idx_[:,None,:].expand(-1,x.shape[1],-1))
            input0 = jt.max(input, 2)
            outputs, states = self.lstm(input0.unsqueeze(1), states)  #Bx1x128
            outputs = outputs.permute(1,0,2)
            p = jt.bmm(outputs,x) #Bx1x1024
            p = nn.softmax(p,dim=-1)
            feat = jt.bmm(origin_x, p.permute(0,2,1))
            all_joints[:,i,:] = feat.squeeze(-1)
        return all_joints
