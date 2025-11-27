import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import json

from finetune_dataset import FineTuneThermalDataset
from model import AnomalyAutoEncoder

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def finetune_autoencoder(
    feedback_data_root,
    weights_path,
    output_dir='output',
    batch_size=8,
    epochs=10,
    learning_rate=1e-4,
    img_size=256,
    device=None,
    latest_folder=None,
    old_multiplier=5,
    val_split=0.2
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and split dataset
    dataset = FineTuneThermalDataset(
        root_dir=feedback_data_root,
        img_size=img_size,
        latest_folder=latest_folder,
        old_multiplier=old_multiplier
    )
    if len(dataset) == 0:
        raise ValueError("No images found for fine-tuning! Check your dataset path.")
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows fix
        pin_memory=True if device.type == 'cuda' else False
    )

    # Gather dataset stats for logging
    latest_count = getattr(dataset, 'latest_count', None)
    old_count = getattr(dataset, 'old_count', None)
    total_count = len(dataset)
    latest_folder_name = getattr(dataset, 'latest_folder_name', None)
    folder_image_counts = getattr(dataset, 'folder_image_counts', None)

    # Initialize model and load weights
    print(f"Loading model weights from {weights_path} ...")
    model = AnomalyAutoEncoder(in_channels=3, latent_dim=128)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Evaluate before fine-tuning
    print("Evaluating before fine-tuning...")
    old_train_loss = evaluate(model, train_loader, criterion, device)
    old_val_loss = evaluate(model, val_loader, criterion, device)

    # Fine-tuning loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'FineTune Epoch {epoch+1}/{epochs}')
        for images in pbar:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        # Validation loss
        model.eval()
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')
        scheduler.step(val_loss)
        # Save best model based on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, output_dir / 'best_finetuned_model.pth')
            print(f'  -> Saved best finetuned model with val loss: {best_val_loss:.6f}')
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, output_dir / f'finetune_checkpoint_epoch_{epoch+1}.pth')
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, output_dir / 'final_finetuned_model.pth')

    # Evaluate after fine-tuning
    print("Evaluating after fine-tuning...")
    new_train_loss = evaluate(model, train_loader, criterion, device)
    new_val_loss = evaluate(model, val_loader, criterion, device)

    # Plot training/validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fine-tuning Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'finetune_training_loss.png')

    # Log details
    log = {
        'weights_path': str(weights_path),
        'feedback_data_root': str(feedback_data_root),
        'output_dir': str(output_dir),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'img_size': img_size,
        'old_train_loss': old_train_loss,
        'new_train_loss': new_train_loss,
        'old_val_loss': old_val_loss,
        'new_val_loss': new_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'latest_folder': latest_folder_name,
        'latest_count': latest_count,
        'old_count': old_count,
        'total_count': total_count,
        'folder_image_counts': folder_image_counts,
        'val_split': val_split,
        'old_multiplier': old_multiplier,
    }
    with open(output_dir / 'finetune_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nFine-tuning completed! Best val loss: {best_val_loss:.6f}")
    print(f"Logs and models saved in: {output_dir}")
    print(f"\nDataset breakdown:")
    print(f"  Latest folder: {latest_folder_name} | Images: {latest_count}")
    print(f"  Sampled old images: {old_count}")
    print(f"  Total images used: {total_count}")
    if folder_image_counts:
        print("  Folder image counts:")
        for folder, count in folder_image_counts.items():
            print(f"    {folder}: {count}")
    return model, train_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune anomaly autoencoder with user feedback.')
    parser.add_argument('--feedback-data', type=str, default='Local_Dataset')
    parser.add_argument('--weights', type=str, default='ML_analysis/models/best_model.pth')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--latest-folder', type=str, default=None, help='Optionally specify latest folder (e.g. 07_2025)')
    parser.add_argument('--old-multiplier', type=int, default=5, help='Multiplier for old data sampling (default 5)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of data for validation (default 0.2)')
    args = parser.parse_args()
    finetune_autoencoder(
        feedback_data_root=args.feedback_data,
        weights_path=args.weights,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        latest_folder=args.latest_folder,
        old_multiplier=args.old_multiplier,
        val_split=args.val_split
    )
