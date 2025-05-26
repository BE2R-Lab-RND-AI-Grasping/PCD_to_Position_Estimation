import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
from pathlib import Path

def rot6d_to_matrix_numpy(rot_6d):
    # Input: shape (6)
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]

    # Normalize first vector
    b1 = a1 / np.linalg.norm(a1)

    # Make second vector orthogonal to b1
    dot = np.sum(b1 * a2,)
    a2_ortho = a2 - dot * b1
    b2 = a2_ortho / np.linalg.norm(a2_ortho)

    # Third basis vector via cross product
    b3 = np.cross(b1, b2)

    # Stack into rotation matrix
    rot_matrix = np.stack([b1, b2, b3], axis=-1)  # shape (3, 3)
    return rot_matrix

def rot6d_to_matrix(x):
    # x: (B, 6) tensor
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    # Normalize first vector
    b1 = torch.nn.functional.normalize(a1, dim=1)

    # Orthogonalize a2 against b1
    dot = torch.sum(b1 * a2, dim=1, keepdim=True)
    a2_ortho = a2 - dot * b1
    b2 = torch.nn.functional.normalize(a2_ortho, dim=1)

    # Third vector via cross product
    b3 = torch.cross(b1, b2, dim=1)

    # Stack vectors as rotation matrix columns
    return torch.stack((b1, b2, b3), dim=-1)  # shape (B, 3, 3)

def geodesic_loss(R_pred, R_gt):
    # R_pred, R_gt: (B, 3, 3)
    RT = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = RT[:, 0, 0] + RT[:, 1, 1] + RT[:, 2, 2]
    cos = (trace - 1) / 2
    cos = torch.clamp(cos, -0.9999, 0.9999)  # Numerical stability
    return torch.mean(torch.acos(cos))

def geodesic_loss_numpy(R_pred, R_gt):
    # R_pred and R_gt are (3, 3)
    RT = R_pred.T @ R_gt
    # RT = np.matmul(np.transpose(R_pred, (0, 2, 1)), R_gt)
    trace = np.trace(RT)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def save_checkpoint(state, filename="new_run/checkpoint.pth"):
    torch.save(state, filename)

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n_samples_val = len(val_loader.dataset)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch_size = batch[0].shape[0]
            point_clouds = batch[0].to(device) # (B, N, 3)
            gt_rotation_6d = batch[1][1].to(device)  # (B, 6)
            gt_rotations = rot6d_to_matrix(gt_rotation_6d)  # (B, 3, 3)
            pred_rotation_6d = model(None, point_clouds)
            pred_rotations = rot6d_to_matrix(pred_rotation_6d)  # (B, 3, 3)
            rot_loss = geodesic_loss(pred_rotations, gt_rotations)
            total_loss += rot_loss.item() * batch_size

    print(f"Validation - Loss: {total_loss/n_samples_val:.4f} | ")

    return total_loss/n_samples_val


def train_model(model, train_loader, val_loader, optimizer, scheduler=None, device=torch.device("cpu"), epochs=100, directory="new_run"):
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter()
    best_val_loss = float('inf')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = len(train_loader.dataset)
        for batch in tqdm(train_loader, desc="Training"):
            batch_size = batch[0].shape[0]
            point_clouds = batch[0].to(device) # (B, N, 3)
            gt_rotation_6d = batch[1][1].to(device)  # (B, 6)
            gt_rotations = rot6d_to_matrix(gt_rotation_6d)  # (B, 3, 3)
            optimizer.zero_grad()
            pred_rotation_6d = model(None, point_clouds)
            pred_rotations = rot6d_to_matrix(pred_rotation_6d)  # (B, 3, 3)
            rot_loss = geodesic_loss(pred_rotations, gt_rotations)
            rot_loss.backward()
            optimizer.step()
            total_loss += rot_loss.item() * batch_size

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Total Loss: {total_loss/n_samples:.4f}")
        
        writer.add_scalar("Loss/Train_Total", total_loss/n_samples, epoch)

        # val_loss = validate_model(model, val_loader, device)
        val_loss = validate_model(
            model, val_loader, device)

        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        # Optional: save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), directory+"/best_model.pth")

        if (epoch+1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, filename=directory+f'/checkpoint_epoch_{epoch+1}.pth')
