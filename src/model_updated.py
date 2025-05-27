import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_utils import downsample_fps, ball_query


class SetAbstactionBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, radius, k, use_xyz=False):
        super().__init__()
        if use_xyz:
            input_dim = input_dim+6
        else:
            input_dim = input_dim+3

        self.radius = radius
        self.k = k
        self.use_xyz = use_xyz

        self.conv = nn.Sequential()
        for out_dim in mlp_dim[:-1]:
            block = nn.Sequential(
                nn.Conv2d(input_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.GELU(),
            )
            self.conv.append(block)
            input_dim = out_dim
        self.conv.append(nn.Conv2d(input_dim, mlp_dim[-1], 1, bias=False))
        self.last_norm = nn.BatchNorm1d(mlp_dim[-1])

    def point_gather(self, src_x, src_xyz, xyz, radius, k):
        # xyz (b, m, 3)
        group_idxs = ball_query(src_xyz, xyz, radius, k)  # (B, m, k)
        # src_xyz (b, n, 3)
        group_idxs = group_idxs.flatten(1, 2)  # (B, m*k)
        grouped_xyz = src_xyz.gather(
            1, group_idxs.unsqueeze(-1).repeat(1, 1, 3))  # (B, m*k, 3)
        grouped_xyz = grouped_xyz.view(
            # (B, m, k, 3)
            grouped_xyz.shape[0], grouped_xyz.shape[1]//k, k, 3)

        grouped_xyz_centered = grouped_xyz - xyz.unsqueeze(2)  # (B, m, k, 3)

        if src_x is not None:
            grouped_x = src_x.gather(
                1, group_idxs.unsqueeze(-1).repeat(1, 1, src_x.shape[-1]))
            grouped_x = grouped_x.view(
                # (B, m, k, C)
                grouped_x.shape[0], grouped_x.shape[1]//k, k, src_x.shape[-1])
            if self.use_xyz:
                # (B, m, k, 6+C)
                out = torch.cat(
                    [grouped_xyz, grouped_xyz_centered, grouped_x], dim=-1)
            else:
                # (B, m, k, 3+C)
                out = torch.cat([grouped_xyz_centered, grouped_x], dim=-1)
        else:
            if self.use_xyz:
                # (B, m, k, 6)
                out = torch.cat([grouped_xyz, grouped_xyz_centered], dim=-1)
            else:
                out = grouped_xyz_centered  # (B, m, k, 3)

        out = out.permute(0, 3, 1, 2)
        return out

    def forward(self, src_x, src_xyz, xyz):
        x = self.point_gather(src_x, src_xyz, xyz, self.radius, self.k)
        x = self.conv(x)
        x = x.max(-1)[0]
        x = F.gelu(self.last_norm(x))
        return x.permute(0, 2, 1)  # (B, m, C)


class PointNet2Backbone(nn.Module):
    def __init__(self, input_dim=0, sa_mlps=[[16, 16, 16], [16, 32, 64]], mlp=[64, 128, 128],downsample_points=[256, 64], radii=[0.1, 0.15], ks=[16, 32], add_xyz=False, emb_mode=False):
        super().__init__()
        self.emb_mode = emb_mode
        self.downsample_points = downsample_points
        self.add_xyz = add_xyz
        self.sa1 = SetAbstactionBlock(
            input_dim=input_dim, mlp_dim=sa_mlps[0], radius=radii[0], k=ks[0], use_xyz=add_xyz)
        self.sa2 = SetAbstactionBlock(
            input_dim=sa_mlps[0][2], mlp_dim=sa_mlps[1], radius=radii[1], k=ks[1], use_xyz=add_xyz)
        scale = 2
        if add_xyz:
            self.global_sa = nn.Sequential(
                # nn.Conv1d(sa_mlps[1][2]+3, mlp[0], 1, bias=False),
                nn.Conv1d(3, mlp[0], 1, bias=False),
                nn.BatchNorm1d(mlp[0]),
                nn.GELU(),
                nn.Conv1d(mlp[0], mlp[1], 1, bias=False),
                nn.BatchNorm1d(mlp[1]),
                nn.GELU(),
                nn.Conv1d(mlp[1], mlp[2], 1, bias=False),
            )
        else:
            self.global_sa = nn.Sequential(
                nn.Conv1d(sa_mlps[1][2], mlp[0], 1, bias=False),
                nn.BatchNorm1d(mlp[0]),
                nn.GELU(),
                nn.Conv1d(mlp[0], mlp[1], 1, bias=False),
                nn.BatchNorm1d(mlp[1]),
                nn.GELU(),
                nn.Conv1d(mlp[1], mlp[2], 1, bias=False),
            )

    def forward(self, x, xyz):

        xyz_1 = downsample_fps(xyz, self.downsample_points[0])
        x1 = self.sa1(x, xyz, xyz_1)  # (B, 512, 128)

        xyz_2 = downsample_fps(xyz_1, self.downsample_points[1])
        x2 = self.sa2(x1, xyz_1, xyz_2)  # (B, 128, 256)
        if self.emb_mode: return torch.cat([x2, xyz_2], dim=-1)
        if self.add_xyz:
            # x2_with_xyz = torch.cat([x2, xyz_2], dim=-1) # (B,  128, 256+3)
            x2_with_xyz = xyz_2 # (B,  128, 256+3)
            x3 = self.global_sa(x2_with_xyz.permute(0, 2, 1))  # (B, 1024, 1)
        else:
            x3 = self.global_sa(x2.permute(0, 2, 1))  # (B, 1024, 1)
        out = x3.max(-1)[0]
        return out


class PointNet2Classification(nn.Module):
    """Model for point cloud classification and position prediction."""

    def __init__(self, num_classes, mlp=[64,32], backbone_params=None, head_norm=True, dropout=0.3, emb_mode=False):
        """Initialize PointNet++ backbone and classification head. 

        Args:
            num_classes (int): number of possible classes
        """
        super().__init__()
        self.emb_mode = emb_mode
        if backbone_params is None:
            self.backbone = PointNet2Backbone(emb_mode=emb_mode)
            last_backbone_layer = 128
        else:
            self.backbone = PointNet2Backbone(**backbone_params, emb_mode=emb_mode)
            last_backbone_layer = backbone_params['mlp'][-1]
        scale = 2
        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(last_backbone_layer)
        self.classification_head = nn.Sequential(
            nn.Linear(last_backbone_layer, mlp[0]),
            norm(mlp[0]),          # Input feature size = 512
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(mlp[0], mlp[1]),
            norm(mlp[1]),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp[1], num_classes)            # 10 output classes
        )

    def forward(self, x, xyz):

        pcd_features = self.backbone(x, xyz)  # (B, 1024)
        if self.emb_mode: return pcd_features
        head_input = F.gelu(self.norm(pcd_features))
        class_logits = self.classification_head(head_input)  # (B, num_classes)

        return class_logits


class PointNet2Translation(nn.Module):
    """Model for point cloud classification and position prediction."""

    def __init__(self,  mlp=[64,32], backbone_params=None, head_norm=True, dropout=0.3):
        """Initialize PointNet++ backbone and classification head. 

        Args:
            num_classes (int): number of possible classes
        """
        super().__init__()
        if backbone_params is None:
            last_backbone_layer = 128
            self.backbone = PointNet2Backbone(add_xyz=True)
        else:
            last_backbone_layer = backbone_params['mlp'][-1]
            self.backbone = PointNet2Backbone(**backbone_params)
        scale = 2
        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(last_backbone_layer)
        self.classification_head = nn.Sequential(
            nn.Linear(last_backbone_layer, mlp[0]),
            norm(mlp[0]),          # Input feature size = 512
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(mlp[0], mlp[1]),
            norm(mlp[1]),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp[1], 3)            # 10 output classes
        )

    def forward(self, x, xyz):
        pcd_features = self.backbone(x, xyz)  # (B, 1024)
        head_input = F.gelu(self.norm(pcd_features))
        translation = self.classification_head(head_input)  # (B, num_classes)

        return translation
    


class PointNet2Rotation(nn.Module):
    """Model for point cloud classification and position prediction."""

    def __init__(self,  mlp=[64,32], backbone_params=None, head_norm=True, dropout=0.3):
        """Initialize PointNet++ backbone and classification head. 

        Args:
            num_classes (int): number of possible classes
        """
        super().__init__()
        if backbone_params is None:
            last_backbone_layer = 128
            self.backbone = PointNet2Backbone(add_xyz=True)
        else:
            last_backbone_layer = backbone_params['mlp'][-1]
            self.backbone = PointNet2Backbone(**backbone_params)
        scale = 2
        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(last_backbone_layer)
        self.classification_head = nn.Sequential(
            nn.Linear(last_backbone_layer, mlp[0]),
            norm(mlp[0]),          # Input feature size = 512
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(mlp[0], mlp[1]),
            norm(mlp[1]),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp[1], 6)            # 10 output classes
        )

    def forward(self, x, xyz):
        pcd_features = self.backbone(x, xyz)  # (B, 1024)
        head_input = F.gelu(self.norm(pcd_features))
        translation = self.classification_head(head_input)  # (B, num_classes)

        return translation