"""Module contains the utility functions to process point clouds for PointNet++.

It includes functions for farthest point sampling, ball query, and indexing points.
"""

import torch

torch.Tensor 


def farthest_point_sample(xyz, n_point, start_idx=None):
    """ Sample npoint farthest points from xyz. 

    Args:
        xyz (torch.Tensor(B, N, 3)): input point cloud coordinates
        npoint (int): number of points to sample

    Returns:
        torch.Tensor(B, npoint): indices of sampled points
    """
    B, N, _ = xyz.shape # N is the number of points in the point cloud
    assert n_point <= N, "not enough points to sample"

    if n_point == N:
        # I need to return all points indices => 0, 1, ..., N-1 but with butch dimension
        return (torch.arange(n_point, dtype=torch.long, device=xyz.device)).repeat(B, 1)
    
    if start_idx is not None:
        # create a tensor of size (B, n_point) with the same value start_idx in all elements
        sampled_idxs = torch.full((B, n_point), start_idx, dtype=torch.long, device=xyz.device)
    else:
        sampled_idxs = torch.randint(N, (B, n_point), dtype=torch.long, device=xyz.device)
    # first point from each batch
    current_point = (xyz[torch.arange(B), sampled_idxs[:, 0]]).unsqueeze(1)  # (B, 1, 3)
    min_dists = torch.full((B, N), dtype=xyz.dtype, device=xyz.device, fill_value=float('inf'))
    for i in range(1, n_point):
        # update distance  
        # dists = torch.linalg.norm(xyz - current_point, dim=-1)
        dists = torch.sum((xyz - current_point) ** 2, -1) # (B, N)
        # for each point we need the minimum distance to the sampled points, and it should be done for each batch
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        idx_farthest = torch.max(min_dists, dim=-1).indices
        sampled_idxs[:, i] = idx_farthest
        current_point[:, 0, :] = xyz[torch.arange(B), idx_farthest]

    return sampled_idxs # (B, npoint)

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

def downsample_fps(xyz:torch.Tensor, n_sample):
    # xyz: (b, n, 3)
    sample_ind = farthest_point_sample(xyz, n_sample, start_idx=0)  # (b, n_sample)

    sample_xyz = xyz.gather(1, sample_ind.unsqueeze(-1).repeat(1,1,3))  # (b, n_sample, 3)
    return sample_xyz


def ball_query(src, query, radius, k):
    # src: (b, n, 3)
    # query: (b, m, 3)
    b, n = src.shape[:2]
    m = query.shape[1]
    dists = torch.cdist(query, src)  # (b, m, n)
    # get the mask of distances greater than radius (True if greater)
    mask = dists > radius ** 2
    # Set invalid distances to large number so they are sorted to the end
    dists[mask] = float('inf')
    # Get the indices of the k nearest neighbors for each point in query
    idx = dists.argsort()[:, :, :k]  # (b, m, k)
    # get the corresponding values from the mask 
    grouped_mask = torch.gather(mask, 2, idx)  # (b, m, k)
    # for each point in query get the first index and repeat it for k times
    first_idx = idx[:, :, 0].unsqueeze(-1).repeat(1, 1, k)  # (b, m, k)
    # Replace invalid entries with first valid index
    idx = torch.where(grouped_mask, first_idx, idx) # (B, m, nsample)

    return idx # (B, m, nsample)

def knn_point(src, query,  k):
    """
    Input:
        k: int
        xyz: (b, n, 3) - src points
        new_xyz: (B, m, 3) - query points
    Return:
        group_idx: (B, m, k) - indices of k nearest neighbors
    """
    # Compute squared distances between new_xyz and xyz
    dists = torch.cdist(query, src)  # (b, m, n)
    # dist = square_distance(xyz, new_xyz)  # (B, npoint, N)

    # Get indices of k smallest distances
    _, group_idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=False)  # (B, m, k)
    return group_idx



# def index_points(points, idx):
#     """Convert indices into points. 

#     Args:
#         points (torch.Tensor(B, N, C)): original point cloud
#         idx (torch.Tensor(B, ...)): indices to gather, shape could be (B, S) or (B, S, K)

#     Returns:
#         torch.Tensor(B, ..., C): gathered points
#     """
#     B = points.shape[0]

#     # Expand batch indices to match shape of idx
#     batch_indices = torch.arange(B, dtype=torch.long).to(points.device)
#     batch_indices = batch_indices.view(
#         B, *([1] * (idx.dim() - 1))).expand_as(idx)

#     return points[batch_indices, idx]

def get_features_by_index(feature_tensor:torch.Tensor, idx_tensor, dim, size):
    if len(idx_tensor.shape) == len(feature_tensor.shape):
        return feature_tensor.gather(dim, idx_tensor)
    else:
        expand_list = ([-1]* len(idx_tensor.shape)).insert(dim, size)
        idx_tensor_expanded = idx_tensor.expand(expand_list)
        return feature_tensor.gather(dim, idx_tensor_expanded)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
torch.Tensor.expand
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
    
    group_idx = ball_query(radius, nsample, xyz, new_xyz) # (B, npoint, nsample)
    # group_idx = knn_point(nsample, xyz, new_xyz)
    
    grouped_xyz = index_points(xyz, group_idx) # (B, npoint, nsample, 3)

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
    


# def sample_and_group_relative(npoint, nsample, xyz, points, returnfps=False, coord3=True):
#     """
#     Perform sampling, grouping, and normalization.

#     Args:
#         npoint (int): number of sampled centers
#         nsample (int): number of neighbors per center
#         xyz (torch.Tensor(B, N, 3)): input coordinates
#         points (Optional[torch.Tensor(B, N, C)]): input features
#         returnfps (bool): whether to return FPS indices

#     Returns:
#         torch.Tensor(B, npoint, 3): sampled center coordinates
#         torch.Tensor (B, npoint, nsample, C+6): grouped points coordinates and features
#         (optionally) torch.Tensor(B, npoint): indices of sampled points
#     """
#     B, N, _ = xyz.shape
#     # 1. Farthest point sampling
#     fps_idx = farthest_point_sample(xyz, npoint) # (B, npoint)
#     # new point cloud built from only the sampled points
#     new_xyz = index_points(xyz, fps_idx)         # (B, npoint, 3)

#     # 2. KNN query
#     # (B, npoint, nsample)
#     group_idx = knn_point(nsample, xyz, new_xyz)
#     # (B, npoint, nsample, 3)
#     grouped_xyz = index_points(xyz, group_idx)

#     # 3. Normalize neighborhoods
#     grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # center at origin

#     if points is not None:
#         grouped_points = index_points(
#             points, group_idx)   # (B, npoint, nsample, C)
#         # (B, npoint, nsample, C+3)
#         if coord3:
#             new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
#         else:
#             new_points = torch.cat([grouped_xyz, grouped_xyz_norm, grouped_points], dim=-1)
#     else:
#         if coord3:
#             new_points = grouped_xyz_norm
#         else:
#             new_points = torch.cat([grouped_xyz, grouped_xyz_norm], dim=-1)

#     if returnfps:
#         return new_xyz, new_points, fps_idx
#     else:
#         return new_xyz, new_points
