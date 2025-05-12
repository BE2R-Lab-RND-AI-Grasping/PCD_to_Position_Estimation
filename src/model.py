
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_utils import sample_and_group


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels, group_all):
        """
        Args:
            npoint (int): number of sampled points
            radius (float): ball query radius
            nsample (int): max number of neighbors in a group
            in_channels (int): input feature dim (excluding coordinates)
            mlp_dim (list[int]): output dimensions of MLP layers
            group_all (bool): whether to process entire point set as one group (for global SA)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channels + 6  # +6 for absolute and relative xyz
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp_channels:
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
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)# just zeroes it is not supposed to be used
            # grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            mean_xyz = torch.mean(grouped_xyz, dim=-1, keepdim=True)  # (B, 1, N, 3)
            grouped_xyz_norm = grouped_xyz - mean_xyz  # center at mean value
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                new_points = torch.cat([grouped_xyz, grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = torch.cat([grouped_xyz, grouped_xyz_norm], dim=-1)
        else:
            # new_xyz are only sampled points (B, npoint, 3)
            # new points are grouped into (B, npoint, nsample, C+6) and have all the points and their coordinates
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # Transpose to (B, C, npoint, nsample) for Conv2d
        # only new_points are to be pocessed by the MLP
        new_points = new_points.permute(0, 3, 1, 2)

        # GPT generated way for MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # Pool across neighbors (nsample)
        new_points = torch.max(new_points, 3)[0]  # (B, mlp[-1], npoint)

        # Transpose back to (B, npoint, mlp[-1])
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetPPBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channels=0,
            mlp_channels=[64, 64, 128],
            group_all=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=32,
            in_channels=128,  # last feature + new xyz absolute and relative
            mlp_channels=[128, 128, 256],
            group_all=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channels=256,
            mlp_channels=[256, 512, 1024],
            group_all=True
        )

    def forward(self, xyz, features=None):
        # xyz: (B, N, 3) — input point cloud with only coordinates
        B, N, _ = xyz.shape

        # Layer 1
        l1_xyz, l1_features = self.sa1(xyz, features)

        # Layer 2
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)

        # Layer 3 (global)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)

        # Global feature vector
        global_feature = l3_features.squeeze(1)  # (B, 1024)

        return global_feature
    

class PoseWithClassModel(nn.Module):
    def __init__(self, num_classes):

        super().__init__()
        self.backbone = PointNetPPBackbone()

        self.class_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.pose_head = nn.Sequential(
            nn.Linear(1024 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 7)  # x, y, z, qx, qy, qz, qw
        )

    def forward(self, xyz):
        global_feature = self.backbone(xyz)           # (B, 1024)
        class_logits = self.class_head(global_feature)  # (B, num_classes)
        class_probs = F.softmax(class_logits, dim=1)

        combined = torch.cat([global_feature, class_logits], dim=1)  # (B, 1024 + C)
        pose = self.pose_head(combined)  # (B, 7)

        return class_logits, pose