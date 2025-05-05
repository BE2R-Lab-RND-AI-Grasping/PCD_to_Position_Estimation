import torch

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: (B, N, 3)
        npoint: int
    Return:
        centroids: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    sampled_points = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)# random sample an index for each batch

    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    
    for i in range(npoint):
        sampled_points[:, i] = farthest
        current_point = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - current_point) ** 2, -1)              # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return sampled_points

def square_distance(src, dst):
    """
    Compute pairwise squared distance between src and dst.
    src: (B, N, 3), dst: (B, M, 3)
    return: (B, M, N)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(dst, src.transpose(1, 2))  # (B, M, N)
    dist += torch.sum(dst ** 2, -1).unsqueeze(2)         # (B, M, 1)
    dist += torch.sum(src ** 2, -1).unsqueeze(1)         # (B, 1, N)
    return dist


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Group local neighborhoods by radius around sampled center points.

    Inputs:
        radius: float
        nsample: int
        xyz: (B, N, 3)
        new_xyz: (B, npoint, 3)
    
    Output:
        group_idx: (B, npoint, nsample) — indices of neighbors
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Compute squared distances from centers to all points
    dist = square_distance(xyz, new_xyz)  # (B, S, N)

    # Mask out points beyond radius
    mask = dist > radius ** 2  # (B, S, N)
    dist[mask] = 1e10

    # Get indices of the closest nsample points per center
    group_idx = dist.argsort()[:, :, :nsample]  # (B, S, nsample)

    # Handle case where fewer than nsample are within radius:
    # Replace masked indices with the first index to avoid out-of-bounds
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    group_idx[mask.sum(-1) < nsample] = group_first[mask.sum(-1) < nsample]

    return group_idx


def index_points(points, idx):
    """
    Index points using batch-wise indices.

    Args:
        points: (B, N, C) — original point cloud
        idx: (B, ...) — indices to gather, shape could be (B, S) or (B, S, K)

    Returns:
        new_points: (B, ..., C) — gathered points
    """
    B = points.shape[0]

    # Expand batch indices to match shape of idx
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device)
    batch_indices = batch_indices.view(B, *([1] * (idx.dim() - 1))).expand_as(idx)

    return points[batch_indices, idx]


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Perform sampling, grouping, and normalization.

    Args:
        npoint: int — number of sampled centers
        radius: float — ball radius
        nsample: int — number of neighbors per center
        xyz: (B, N, 3) — input coordinates
        points: (B, N, C) or None — input features
        returnfps: bool — whether to return FPS indices

    Returns:
        new_xyz: (B, npoint, 3) — sampled center coordinates
        new_points: (B, npoint, nsample, C+3) — grouped and normalized features
        (optionally) fps_idx: (B, npoint)
    """
    B, N, _ = xyz.shape

    # 1. Farthest point sampling
    fps_idx = farthest_point_sample(xyz, npoint)         # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)                 # (B, npoint, 3)

    # 2. Ball query
    group_idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, group_idx)             # (B, npoint, nsample, 3)

    # 3. Normalize neighborhoods
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # center at origin

    if points is not None:
        grouped_points = index_points(points, group_idx)   # (B, npoint, nsample, C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
    else:
        new_points = grouped_xyz_norm  # no features, only coords

    if returnfps:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        Args:
            npoint: int — number of sampled points
            radius: float — ball query radius
            nsample: int — max number of neighbors
            in_channel: int — input feature dim (excluding xyz)
            mlp: list[int] — output dimensions of MLP layers
            group_all: bool — whether to process entire point set as one group (for global SA)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel + 3  # +3 for relative xyz
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C) or None

        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, mlp[-1])
        """
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # Transpose to (B, C, npoint, nsample) for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # Pool across neighbors (nsample)
        new_points = torch.max(new_points, 3)[0]  # (B, mlp[-1], npoint)

        # Transpose back to (B, npoint, mlp[-1])
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points

