"""Module contains the utility functions to process point clouds for PointNet++.

It includes functions for farthest point sampling, ball query, and indexing points.
"""

import torch

torch.Tensor 
def farthest_point_sample(xyz, npoint):
    """ Sample npoint farthest points from xyz. 

    Args:
        xyz (torch.Tensor(B, N, 3)): input point cloud coordinates
        npoint (int): number of points to sample

    Returns:
        torch.Tensor(B, npoint): indices of sampled points
    """
    B, N, _ = xyz.shape # N is the number of points in the point cloud
    # tensor for sampled points
    sampled_points = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    # initial distances 
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    # sample a random index for each batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)  
    # explicit indexing to ensure single farthest point per batch
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    # at each step we are looking for a point which is farthest from the sampled points
    for i in range(npoint):
        sampled_points[:, i] = farthest # farthest is the index 
        current_point = xyz[batch_indices,
                            farthest, :].unsqueeze(1)  # (B, 1, 3) of points
        dist = torch.sum((xyz - current_point) ** 2, -1) # (B, N)
        # adding new point we update distance by points that are closer to it then to any of the previous points
        mask = dist < distance
        distance[mask] = dist[mask]
        # the above code process the previous point to sample a new one in this line 
        farthest = torch.max(distance, -1)[1]

    return sampled_points


def square_distance(src, dst):
    """Compute pairwise squared distance between src and dst.

    Args:
        src (torch.Tensor(B, N, 3)): source points
        dst (torch.Tensor(B, M, 3)): destination points

    Returns: 
        torch.Tensor(B, M, N): pairwise squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(dst, src.transpose(1, 2))  # (B, M, N)
    dist += torch.sum(dst ** 2, -1).unsqueeze(2)         # (B, M, 1)
    dist += torch.sum(src ** 2, -1).unsqueeze(1)         # (B, 1, N)
    return dist

def knn_point(k, xyz, new_xyz):
    """
    Input:
        k: int
        xyz: (B, N, 3) - full point cloud
        new_xyz: (B, npoint, 3) - query points
    Return:
        group_idx: (B, npoint, k) - indices of k nearest neighbors
    """
    # Compute squared distances between new_xyz and xyz
    dist = square_distance(xyz, new_xyz)  # (B, npoint, N)

    # Get indices of k smallest distances
    _, group_idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)  # (B, npoint, k)
    return group_idx


def ball_query(radius, nsample, xyz, new_xyz):
    """Group local neighborhoods by radius around sampled center points.

    Args:
        radius (float): radius of the ball query
        nsample (int): number of neighbors in each group
        xyz (torch.Tensor(B, N, 3)): input point cloud coordinates
        new_xyz (B, npoint, 3): sampled center points

    Returns:
        torch.Tensor(B, npoint, nsample): indices of neighbors
    """

    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    # Compute squared distance between each new_xyz and all xyz
    dists = square_distance(xyz, new_xyz)  # (B, S, N)
    # Mask: True where distance > radius^2 => invalid
    mask = dists > radius ** 2
    # Set invalid distances to large number so they are sorted to the end
    dists[mask] = 1e10
    # Find indices of the nearest nsample neighbors (even if not enough valid)
    idx = dists.argsort()[:, :, :nsample]  # (B, S, nsample)
    # Gather distance mask for top-k neighbors
    grouped_mask = torch.gather(mask, 2, idx)  # (B, S, nsample)
    # Get the first valid index in each group to use for padding
    first_idx = idx[:, :, 0].unsqueeze(-1).expand(-1, -1, nsample)  # (B, S, nsample)
    # Replace invalid entries with first valid index
    idx = torch.where(grouped_mask, first_idx, idx) # (B, S, nsample)

    return idx


def index_points(points, idx):
    """Convert indices into points. 

    Args:
        points (torch.Tensor(B, N, C)): original point cloud
        idx (torch.Tensor(B, ...)): indices to gather, shape could be (B, S) or (B, S, K)

    Returns:
        torch.Tensor(B, ..., C): gathered points
    """
    B = points.shape[0]

    # Expand batch indices to match shape of idx
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device)
    batch_indices = batch_indices.view(
        B, *([1] * (idx.dim() - 1))).expand_as(idx)

    return points[batch_indices, idx]


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Perform sampling, grouping, and normalization.

    Args:
        npoint (int): number of sampled centers
        radius (float): ball radius
        nsample (int): number of neighbors per center
        xyz (torch.Tensor(B, N, 3)): input coordinates
        points (Optional[torch.Tensor(B, N, C)]): input features
        returnfps (bool): whether to return FPS indices

    Returns:
        torch.Tensor(B, npoint, 3): sampled center coordinates
        torch.Tensor (B, npoint, nsample, C+6): grouped points coordinates and features
        (optionally) torch.Tensor(B, npoint): indices of sampled points
    """
    B, N, _ = xyz.shape
    # 1. Farthest point sampling
    fps_idx = farthest_point_sample(xyz, npoint) # (B, npoint)
    # new point cloud built from only the sampled points
    new_xyz = index_points(xyz, fps_idx)         # (B, npoint, 3)

    # 2. Ball query
    # (B, npoint, nsample)
    group_idx = ball_query(radius, nsample, xyz, new_xyz)
    # group_idx = knn_point(nsample, xyz, new_xyz)
    # (B, npoint, nsample, 3)
    grouped_xyz = index_points(xyz, group_idx)

    # 3. Normalize neighborhoods
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # center at origin

    if points is not None:
        grouped_points = index_points(
            points, group_idx)   # (B, npoint, nsample, C)
        # (B, npoint, nsample, C+6)
        new_points = torch.cat([grouped_xyz, grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = torch.cat([grouped_xyz, grouped_xyz_norm], dim=-1)

    if returnfps:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points
    


def sample_and_group_relative(npoint, nsample, xyz, points, returnfps=False, coord3=True):
    """
    Perform sampling, grouping, and normalization.

    Args:
        npoint (int): number of sampled centers
        nsample (int): number of neighbors per center
        xyz (torch.Tensor(B, N, 3)): input coordinates
        points (Optional[torch.Tensor(B, N, C)]): input features
        returnfps (bool): whether to return FPS indices

    Returns:
        torch.Tensor(B, npoint, 3): sampled center coordinates
        torch.Tensor (B, npoint, nsample, C+6): grouped points coordinates and features
        (optionally) torch.Tensor(B, npoint): indices of sampled points
    """
    B, N, _ = xyz.shape
    # 1. Farthest point sampling
    fps_idx = farthest_point_sample(xyz, npoint) # (B, npoint)
    # new point cloud built from only the sampled points
    new_xyz = index_points(xyz, fps_idx)         # (B, npoint, 3)

    # 2. KNN query
    # (B, npoint, nsample)
    group_idx = knn_point(nsample, xyz, new_xyz)
    # (B, npoint, nsample, 3)
    grouped_xyz = index_points(xyz, group_idx)

    # 3. Normalize neighborhoods
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # center at origin

    if points is not None:
        grouped_points = index_points(
            points, group_idx)   # (B, npoint, nsample, C)
        # (B, npoint, nsample, C+3)
        if coord3:
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = torch.cat([grouped_xyz, grouped_xyz_norm, grouped_points], dim=-1)
    else:
        if coord3:
            new_points = grouped_xyz_norm
        else:
            new_points = torch.cat([grouped_xyz, grouped_xyz_norm], dim=-1)

    if returnfps:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points
