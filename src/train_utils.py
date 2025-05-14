import torch
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
    n_batches = len(val_loader)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
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

            total_raw_cls_loss += cls_loss.item()
            total_raw_trans_loss+= trans_loss.item()
            total_raw_rot_loss += rot_loss.item()
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

            # total_loss += loss.item()
            # total_cls_loss += (var_cls * cls_loss).item()
            # total_pose_loss += (var_pos * trans_loss + var_ori * rot_loss).item()

            total_loss += loss.item()
            total_cls_loss += (var_cls * cls_loss).item()
            total_trans_loss += (var_pos * trans_loss).item()
            total_rot_loss += (var_ori * rot_loss).item()

            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_pose_loss += pose_loss.item()

            preds = class_logits.argmax(dim=1)
            correct += (preds == class_labels).sum().item()
            total += class_labels.size(0)

    accuracy = correct / total * 100

    print(f"Validation - Loss: {total_loss/n_batches:.4f} | "
          f"Cls Loss: {total_raw_cls_loss/n_batches:.4f} | "
          f"Trans Loss: {total_raw_trans_loss/n_batches:.4f} | "
          f"Rot Loss: {total_raw_rot_loss/n_batches:.4f} | "
          f"Accuracy: {accuracy:.2f}%")

    return total_loss/n_batches, total_cls_loss / n_batches, total_trans_loss / n_batches, total_rot_loss / n_batches, total_raw_cls_loss/n_batches, total_raw_trans_loss/n_batches,total_raw_rot_loss/n_batches,accuracy


# --- Training function ---
def save_checkpoint(state, filename="new_run/checkpoint.pth"):
    torch.save(state, filename)


def train_model(model, train_loader, val_loader, optimizer, scheduler=None, device=torch.device("cpu"), epochs=100, alpha=1.0, beta=1.0):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    model.to(device)

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
        # total_pose_loss = 0.0
        # print(torch.cuda.get_device_name(0), device)
        n_batches = len(train_loader)
        for batch in tqdm(train_loader, desc="Training"):
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            point_clouds = batch[0].to(device)            # (B, N, 3)
            class_labels = batch[1][0].to(device)       # (B,)
            gt_translation = batch[1][2].to(device)     # (B, 3)
            gt_quaternion = batch[1][1].to(device)       # (B, 4)

            optimizer.zero_grad()

            class_logits, pred_pose = model(point_clouds)
            # print(f"Input device: {point_clouds.device}")
            # print(f"Model device: {next(model.parameters()).device}")
            pred_translation = pred_pose[:, :3]
            pred_quaternion = pred_pose[:, 3:]

            cls_loss = F.cross_entropy(class_logits, class_labels)
            trans_loss = F.mse_loss(pred_translation, gt_translation)
            rot_loss = quaternion_loss(pred_quaternion, gt_quaternion)

            total_raw_cls_loss += cls_loss.item()
            total_raw_trans_loss+= trans_loss.item()
            total_raw_rot_loss += rot_loss.item()

            # Learned weights (convert log variances to actual variances)
            var_cls = torch.exp(-model.log_var_cls)  # 1/(2σ²) = exp(-log_var)
            var_pos = torch.exp(-model.log_var_pos)
            var_ori = torch.exp(-model.log_var_ori)
            
            # Weighted losses + regularization term
            additional_loss = model.log_var_cls + model.log_var_pos + model.log_var_ori
            total_additional_loss += additional_loss.item()
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
            # end.record()
            # torch.cuda.synchronize()
            # print(f"GPU time: {start.elapsed_time(end)} ms")
            # total_loss += loss.item()
            # total_cls_loss += cls_loss.item()
            # total_pose_loss += pose_loss.item()
            total_loss += loss.item()
            total_cls_loss += (var_cls * cls_loss).item()
            total_trans_loss += (var_pos * trans_loss).item()
            total_rot_loss += (var_ori * rot_loss).item()
            # total_pose_loss += (var_pos * trans_loss + var_ori * rot_loss).item()


        print(f"Epoch {epoch+1}/{epochs} - "
              f"Total Loss: {total_loss/n_batches:.4f} - "
              f"Cls: {total_cls_loss/n_batches:.4f} - Trans: {total_trans_loss/n_batches:.4f} - Rot: {total_rot_loss/n_batches:.4f}")
        
        writer.add_scalar("Loss/Train_Total", total_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Classification", total_cls_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Translation", total_trans_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Rotation", total_rot_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Additional", total_additional_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Raw_Classification", total_raw_cls_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Raw_Translation", total_raw_trans_loss/n_batches, epoch)
        writer.add_scalar("Loss/Train_Raw_Rotation", total_raw_rot_loss/n_batches, epoch)

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
