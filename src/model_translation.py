"""Model that predicts object class and position using the point cloud"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_utils import sample_and_group,sample_and_group_relative


class PointNetSetAbstraction(nn.Module):
    """Main module of the PointNet++ that groups points and extract features for each group"""

    def __init__(self, npoint, nsample, in_channels, mlp_channels, group_all, coord3=True):
        """Network initialization and parameter setting.

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
        self.nsample = nsample
        self.group_all = group_all
        self.coord3 = coord3
        dropout_p = 0.3
        if self.coord3:
            last_channel = in_channels + 3  # +3 for relative xyz
        else:
            last_channel = in_channels + 6
        self.mlp = nn.ModuleList()
        for i, out_channel in enumerate(mlp_channels):
            self.mlp.append(nn.Conv2d(last_channel, out_channel, 1))
            # if not self.group_all:
            self.mlp.append(nn.BatchNorm2d(out_channel))
            self.mlp.append(nn.ReLU(inplace=True))
            # self.mlp.append(nn.Dropout2d(p=dropout_p))
            last_channel = out_channel

    def forward(self, xyz, points):
        """Forward network propagation.

        Args:group_all=True
            xyz (torch.Tensor(Bgroup_all=True, N, 3)): point coordinates
            points Optional[torch.Tensor(B, N, C)]: point features

        Returns:
            torch.Tensor(B, npoint, 3): new point cloud
            torch.Tensor(B, npoint, mlp[-1]): new point features
        """

        # group all is true if there is only one group
        if self.group_all:
            # simple replacement for sample_and_group if there is only one group
            # just zeroes it is not supposed to be used
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            mean_xyz = torch.mean(grouped_xyz, dim=-2,keepdim=True)  # (B, 1, N, 3)
            grouped_xyz_norm = grouped_xyz - mean_xyz  # center at mean value
            # grouped_xyz_norm = grouped_xyz # no normalization to keep spacial information
            if points is not None:
                grouped_points = points.view(
                    points.shape[0], 1, points.shape[1], -1)
                
                if self.coord3:
                    new_points = torch.cat(
                        [grouped_xyz, grouped_points], dim=-1)
                else:
                    new_points = torch.cat(
                        [grouped_xyz, grouped_xyz_norm, grouped_points], dim=-1)
                # new_points = torch.cat(
                #     [grouped_xyz_norm, grouped_points], dim=-1)
            else:
                if self.coord3:
                    new_points = grouped_xyz
                else:
                    new_points = torch.cat([grouped_xyz, grouped_xyz_norm], dim=-1)
                # 
        else:
            # new_xyz are only sampled points (B, npoint, 3)
            # new points are grouped into (B, npoint, nsample, C+6) and have all the points and their coordinates
            new_xyz, new_points = sample_and_group_relative(
                self.npoint, self.nsample, xyz, points, coord3=self.coord3)

        # Transpose to (B, C, npoint, nsample) for Conv2d
        # only new_points are to be pocessed by the MLP
        new_points = new_points.permute(0, 3, 1, 2)
        # use 2D convolution to apply same FCL for all points at once

        for layer in self.mlp:
            new_points = layer(new_points)

        # Pool across neighbors (nsample)
        new_points = torch.max(new_points, 3)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetPPBackbone(nn.Module):
    """PontNet++ module for point cloud feature extraction.
    """

    def __init__(self, coord3=True):
        """Initialize several consecutive PointNetSetAbstraction layers for hierarchical feature extraction.

        The last layer should have group_all=True to return the global features of the remaining point cloud.
        """
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=256,  nsample=32,
            in_channels=0,
            mlp_channels=[32, 32, 64],
            group_all=False,coord3=coord3
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,  nsample=16,
            in_channels=64, 
            mlp_channels=[64, 64, 128],
            group_all=False,coord3=coord3
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None,  nsample=None,
            in_channels=128,
            mlp_channels=[128, 256, 512],
            group_all=True,coord3=coord3
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


class TranslationModel(nn.Module):
    """Model for point cloud classification and position prediction."""
    def __init__(self,coord3=True):
        """Initialize PointNet++ backbone and classification head. 

        Args:
            num_classes (int): number of possible classes
        """
        super().__init__()
        self.backbone = PointNetPPBackbone(coord3=coord3)

        self.translation_head = nn.Sequential(
            nn.Linear(512, 128),          # Input feature size = 512
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(64, 3)            # 10 output classes
        )


    def forward(self, xyz):
        global_feature = self.backbone(xyz)           # (B, 1024)
        translation = self.translation_head(global_feature)  # (B, num_classes)

        return translation
