import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

# --- Loss for quaternion ---
def quaternion_loss(pred_q, gt_q):
    pred_q = F.normalize(pred_q, dim=-1)
    gt_q = F.normalize(gt_q, dim=-1)
    dot_product = torch.sum(pred_q * gt_q, dim=-1)
    loss = 1 - torch.abs(dot_product)
    return loss.mean()

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

def validate_model(model, val_loader, device, alpha=1.0, beta=1.0):
    model.eval()
    total_raw_cls_loss = 0.0
    total_raw_trans_loss = 0.0
    total_raw_rot_loss = 0.0

    total_loss = 0.0
    total_cls_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0
    
    correct = 0
    total = 0
    n_samples = len(val_loader.dataset)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch_size = batch[0].shape[0]
            point_clouds = batch[0].to(device)            # (B, N, 3)
            class_labels = batch[1][0].to(device)       # (B,)
            gt_translation = batch[1][2].to(device)     # (B, 3)
            gt_rotation_6d = batch[1][1].to(device)  # (B, 6)
            gt_rotations = rot6d_to_matrix(gt_rotation_6d)  # (B, 3, 3)

            class_logits, pred_pose = model(point_clouds)

            pred_translation = pred_pose[:, :3]
            pred_rotation_6d = pred_pose[:, 3:]
            pred_rotations = rot6d_to_matrix(pred_rotation_6d)  # (B, 3, 3)

            cls_loss = F.cross_entropy(class_logits, class_labels)
            trans_loss = F.mse_loss(pred_translation, gt_translation)
            rot_loss = geodesic_loss(pred_rotations, gt_rotations)

            total_raw_cls_loss += cls_loss.item() * batch_size
            total_raw_trans_loss+= trans_loss.item() * batch_size
            total_raw_rot_loss += rot_loss.item() * batch_size

            # Learned weights (convert log variances to actual variances)
            var_cls = torch.exp(-model.log_var_cls)  # 1/(2σ²) = exp(-log_var)
            var_pos = torch.exp(-model.log_var_pos)
            var_ori = torch.exp(-model.log_var_ori)
            
            # Weighted losses + regularization term
            additional_loss = model.log_var_cls + model.log_var_pos + model.log_var_ori
            loss = (
                var_cls * cls_loss +
                var_pos * trans_loss +
                var_ori * rot_loss +
                additional_loss
            )
            # pose_loss = trans_loss + beta * rot_loss
            # loss = cls_loss + alpha * pose_loss

            # total_loss += loss.item()
            # total_cls_loss += (var_cls * cls_loss).item()
            # total_pose_loss += (var_pos * trans_loss + var_ori * rot_loss).item()

            total_loss += loss.item() * batch_size
            total_cls_loss += (var_cls * cls_loss).item() * batch_size
            total_trans_loss += (var_pos * trans_loss).item() * batch_size
            total_rot_loss += (var_ori * rot_loss).item() * batch_size

            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_pose_loss += pose_loss.item()

            preds = class_logits.argmax(dim=1)
            correct += (preds == class_labels).sum().item()
            total += class_labels.size(0)

    accuracy = correct / total * 100

    print(f"Validation - Loss: {total_loss/n_samples:.4f} | "
          f"Cls Loss: {total_raw_cls_loss/n_samples:.4f} | "
          f"Trans Loss: {total_raw_trans_loss/n_samples:.4f} | "
          f"Rot Loss: {total_raw_rot_loss/n_samples:.4f} | "
          f"Accuracy: {accuracy:.2f}%")

    return total_loss/n_samples, total_cls_loss / n_samples, total_trans_loss / n_samples, total_rot_loss / n_samples, total_raw_cls_loss/n_samples, total_raw_trans_loss/n_samples,total_raw_rot_loss/n_samples,accuracy


# --- Training function ---
def save_checkpoint(state, filename="new_run/checkpoint.pth"):
    torch.save(state, filename)


def train_model(model, train_loader, val_loader, optimizer, scheduler=None, device=torch.device("cpu"), epochs=100, alpha=1.0, beta=1.0):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    model.to(device)
    # for f in model.parameters():
    #     print(f"Parameter: {f}")
    for epoch in range(epochs):
        model.train()
        total_raw_cls_loss = 0.0
        total_raw_trans_loss = 0.0
        total_raw_rot_loss = 0.0

        total_loss = 0.0
        total_cls_loss = 0.0
        total_trans_loss = 0.0
        total_rot_loss = 0.0
        total_additional_loss = 0.0

        n_samples = len(train_loader.dataset)
        for batch in tqdm(train_loader, desc="Training"):
            batch_size = batch[0].shape[0]
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            point_clouds = batch[0].to(device)            # (B, N, 3)
            class_labels = batch[1][0].to(device)       # (B,)
            gt_translation = batch[1][2].to(device)     # (B, 3)
            gt_rotation_6d = batch[1][1].to(device)  # (B, 6)
            gt_rotations = rot6d_to_matrix(gt_rotation_6d)  # (B, 3, 3)
            optimizer.zero_grad()

            if torch.isnan(point_clouds).any():
                print("NaN in point clouds")
                
            if torch.isnan(class_labels).any():
                print("NaN in class labels")
            if torch.isnan(gt_translation).any():
                print("NaN in gt translation")
            if torch.isnan(gt_rotation_6d).any():
                print("NaN in gt rotation 6d")

            class_logits, pred_pose = model(point_clouds)

            pred_translation = pred_pose[:, :3]
            pred_rotation_6d = pred_pose[:, 3:]
            pred_rotations = rot6d_to_matrix(pred_rotation_6d)  # (B, 3, 3)
            

            cls_loss = F.cross_entropy(class_logits, class_labels)
            trans_loss = F.mse_loss(pred_translation, gt_translation)
            rot_loss = geodesic_loss(pred_rotations, gt_rotations)
            # Learned weights (convert log variances to actual variances)
            var_cls = torch.exp(-model.log_var_cls)  # 1/(2σ²) = exp(-log_var)
            var_pos = torch.exp(-model.log_var_pos)
            var_ori = torch.exp(-model.log_var_ori)
            
            # Weighted losses + regularization term
            additional_loss = model.log_var_cls + model.log_var_pos + model.log_var_ori
            
            loss = (
                var_cls * cls_loss +
                var_pos * trans_loss +
                var_ori * rot_loss +
                additional_loss
            )

            # pose_loss = trans_loss + beta * rot_loss
            # loss = cls_loss + alpha * pose_loss
            loss.backward()
            optimizer.step()
            for f in model.parameters():
                if torch.isnan(f).any():
                    print("NaN in model parameters")
                    break
            total_additional_loss += additional_loss.item() * batch_size
            total_raw_cls_loss += cls_loss.item() * batch_size
            total_raw_trans_loss+= trans_loss.item() * batch_size
            total_raw_rot_loss += rot_loss.item() * batch_size
            # print(total_raw_cls_loss, total_raw_trans_loss, total_raw_rot_loss)
            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_pose_loss += pose_loss.item()
            total_loss += loss.item() * batch_size
            total_cls_loss += (var_cls * cls_loss).item() * batch_size
            total_trans_loss += (var_pos * trans_loss).item() * batch_size
            total_rot_loss += (var_ori * rot_loss).item() * batch_size
            # total_pose_loss += (var_pos * trans_loss + var_ori * rot_loss).item()


        print(f"Epoch {epoch+1}/{epochs} - "
              f"Total Loss: {total_loss/n_samples:.4f} - "
              f"Cls: {total_cls_loss/n_samples:.4f} - Trans: {total_trans_loss/n_samples:.4f} - Rot: {total_rot_loss/n_samples:.4f}")
        
        writer.add_scalar("Loss/Train_Total", total_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Classification", total_cls_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Translation", total_trans_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Rotation", total_rot_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Additional", total_additional_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Raw_Classification", total_raw_cls_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Raw_Translation", total_raw_trans_loss/n_samples, epoch)
        writer.add_scalar("Loss/Train_Raw_Rotation", total_raw_rot_loss/n_samples, epoch)

        writer.add_scalar("Weights/cls", model.log_var_cls.item(), epoch)
        writer.add_scalar("Weights/pos", model.log_var_pos.item(), epoch)
        writer.add_scalar("Weights/rot", model.log_var_ori.item(), epoch)

        # val_loss = validate_model(model, val_loader, device)
        val_loss, val_cls_loss, val_trans_loss, val_rot_loss, val_raw_cls_loss, val_raw_trans_loss, val_raw_rot_loss, val_accuracy = validate_model(
            model, val_loader, device)

        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Val_Classification", val_cls_loss, epoch)
        writer.add_scalar("Loss/Val_Translation", val_trans_loss, epoch)
        writer.add_scalar("Loss/Val_Rotation", val_rot_loss, epoch)
        writer.add_scalar("Loss/Val_Raw_Classification", val_raw_cls_loss, epoch)
        writer.add_scalar("Loss/Val_Raw_Translation", val_raw_trans_loss, epoch)
        writer.add_scalar("Loss/Val_Raw_Rotation", val_raw_rot_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        # Optional: save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "new_run/best_model.pth")

        if (epoch+1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, filename=f'new_run/checkpoint_epoch_{epoch+1}.pth')
