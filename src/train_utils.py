import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# --- Loss for quaternion ---
def quaternion_loss(pred_q, gt_q):
    pred_q = F.normalize(pred_q, dim=-1)
    gt_q = F.normalize(gt_q, dim=-1)
    dot_product = torch.sum(pred_q * gt_q, dim=-1)
    loss = 1 - torch.abs(dot_product)
    return loss.mean()


def validate_model(model, val_loader, device, alpha=1.0, beta=1.0):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pose_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            point_clouds = batch[0].to(device)            # (B, N, 3)
            class_labels = batch[1][0].to(device)       # (B,)
            gt_translation = batch[1][2].to(device)     # (B, 3)
            gt_quaternion = batch[1][1].to(device)       # (B, 4)

            class_logits, pred_pose = model(point_clouds)

            pred_translation = pred_pose[:, :3]
            pred_quaternion = pred_pose[:, 3:]

            cls_loss = F.cross_entropy(class_logits, class_labels)
            trans_loss = F.mse_loss(pred_translation, gt_translation)
            rot_loss = quaternion_loss(pred_quaternion, gt_quaternion)

            # Learned weights (convert log variances to actual variances)
            var_cls = torch.exp(-model.log_var_cls)  # 1/(2σ²) = exp(-log_var)
            var_pos = torch.exp(-model.log_var_pos)
            var_ori = torch.exp(-model.log_var_ori)
            
            # Weighted losses + regularization term
            loss = (
                var_cls * cls_loss + 
                var_pos * trans_loss + 
                var_ori * rot_loss + 
                model.log_var_cls + model.log_var_pos + model.log_var_ori
            )
            # pose_loss = trans_loss + beta * rot_loss
            # loss = cls_loss + alpha * pose_loss

            total_loss += loss.item()
            total_cls_loss += (var_cls * cls_loss).item()
            total_pose_loss += (var_pos * trans_loss + var_ori * rot_loss).item()

            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_pose_loss += pose_loss.item()

            preds = class_logits.argmax(dim=1)
            correct += (preds == class_labels).sum().item()
            total += class_labels.size(0)

    accuracy = correct / total * 100
    avg_loss = total_loss / len(val_loader)

    print(f"Validation - Loss: {avg_loss:.4f} | "
          f"Cls Loss: {total_cls_loss:.4f} | "
          f"Pose Loss: {total_pose_loss:.4f} | "
          f"Accuracy: {accuracy:.2f}%")

    return avg_loss, total_cls_loss / len(val_loader), total_pose_loss / len(val_loader), accuracy
# --- Training function ---


def train_model(model, train_loader, val_loader, optimizer, device, epochs=100, alpha=1.0, beta=1.0):
    writer = SummaryWriter(log_dir="runs/pose_classification")
    best_val_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_pose_loss = 0.0

        for batch in train_loader:
            point_clouds = batch[0].to(device)            # (B, N, 3)
            class_labels = batch[1][0].to(device)       # (B,)
            gt_translation = batch[1][2].to(device)     # (B, 3)
            gt_quaternion = batch[1][1].to(device)       # (B, 4)

            optimizer.zero_grad()

            class_logits, pred_pose = model(point_clouds)

            pred_translation = pred_pose[:, :3]
            pred_quaternion = pred_pose[:, 3:]

            cls_loss = F.cross_entropy(class_logits, class_labels)
            trans_loss = F.mse_loss(pred_translation, gt_translation)
            rot_loss = quaternion_loss(pred_quaternion, gt_quaternion)
            # Learned weights (convert log variances to actual variances)
            var_cls = torch.exp(-model.log_var_cls)  # 1/(2σ²) = exp(-log_var)
            var_pos = torch.exp(-model.log_var_pos)
            var_ori = torch.exp(-model.log_var_ori)
            
            # Weighted losses + regularization term
            loss = (
                var_cls * cls_loss + 
                var_pos * trans_loss + 
                var_ori * rot_loss + 
                model.log_var_cls + model.log_var_pos + model.log_var_ori
            )
            # pose_loss = trans_loss + beta * rot_loss
            # loss = cls_loss + alpha * pose_loss

            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_pose_loss += pose_loss.item()
            total_loss += loss.item()
            total_cls_loss += (var_cls * cls_loss).item()
            total_pose_loss += (var_pos * trans_loss + var_ori * rot_loss).item()

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Total Loss: {total_loss:.4f} - "
              f"Cls: {total_cls_loss:.4f} - Pose: {total_pose_loss:.4f}")
        writer.add_scalar("Loss/Train_Total", total_loss, epoch)
        writer.add_scalar("Loss/Train_Classification", total_cls_loss, epoch)
        writer.add_scalar("Loss/Train_Pose", total_pose_loss, epoch)
        writer.add_scalar("Weights/cls", model.log_var_cls.item(), epoch)
        writer.add_scalar("Weights/pos", model.log_var_pos.item(), epoch)
        writer.add_scalar("Weights/rot", model.log_var_ori.item(), epoch)

        # val_loss = validate_model(model, val_loader, device)
        val_loss, val_cls_loss, val_pose_loss, val_accuracy = validate_model(
            model, val_loader, device)

        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Val_Classification", val_cls_loss, epoch)
        writer.add_scalar("Loss/Val_Pose", val_pose_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        # Optional: save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
