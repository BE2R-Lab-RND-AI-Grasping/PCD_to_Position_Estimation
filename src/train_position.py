import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
from pathlib import Path

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
            gt_translation = batch[1][2].to(device) # (B, 3)
            pred_translation = model(None, point_clouds)
            trans_loss = F.l1_loss(pred_translation, gt_translation)
            total_loss += trans_loss.item() * batch_size

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
            gt_translation = batch[1][2].to(device) # (B, 3)
            optimizer.zero_grad()
            pred_translation = model(None, point_clouds)
            trans_loss = F.l1_loss(pred_translation, gt_translation)
            # trans_loss = F.l1_loss(pred_translation, gt_translation)
            trans_loss.backward()
            optimizer.step()
            total_loss += trans_loss.item() * batch_size

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
