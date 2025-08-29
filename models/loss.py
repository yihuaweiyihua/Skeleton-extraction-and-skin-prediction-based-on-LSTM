import jittor as jt

def get_simplification_loss(ref_pc, samp_pc, pc_size = 1000, gamma=1, delta=0):

    # ref_pc and samp_pc are B x N x 3 matrices
    cost_p2_p1 = square_distance(ref_pc, samp_pc).min(-1)
    cost_p1_p2 = square_distance(samp_pc, ref_pc).min(-1)
    max_cost = jt.max(cost_p1_p2, dim=1)
    max_cost = jt.mean(max_cost)
    cost_p1_p2 = jt.mean(cost_p1_p2)
    cost_p2_p1 = jt.mean(cost_p2_p1)
    loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
    return loss

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * jt.matmul(src, dst.permute(0, 2, 1))
    dist += jt.sum(src ** 2, -1).view(B, N, 1)
    dist += jt.sum(dst ** 2, -1).view(B, 1, M)
    return dist

if __name__ == '__main__':
    ref_pc = jt.randn(10, 100, 3)
    samp_pc = jt.randn(10, 10, 3)
    print(get_simplification_loss(ref_pc, samp_pc, 100))