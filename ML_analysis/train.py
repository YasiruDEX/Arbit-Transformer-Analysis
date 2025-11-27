import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from dataset import ThermalDataset
from model import AnomalyAutoEncoder


def train_autoencoder(
    data_root='Dataset',
    batch_size=8,
    epochs=50,
    learning_rate=0.001,
    img_size=256,
    save_dir='models',
    device=None
):
    """
    Train AutoEncoder on normal thermal images only
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset (only normal images)
    print("Loading training data...")
    train_dataset = ThermalDataset(
        root_dir=data_root,
        mode='train',
        img_size=img_size
    )
    
    if len(train_dataset) == 0:
        raise ValueError("No training images found! Check your dataset path.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    model = AnomalyAutoEncoder(in_channels=3, latent_dim=128)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    train_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_dir / 'best_model.pth')
            print(f'  -> Saved best model with loss: {best_loss:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_dir / 'final_model.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_loss.png')
    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    print(f"Models saved in: {save_dir}")
    
    return model, train_losses


if __name__ == '__main__':
    # Train the model
    model, losses = train_autoencoder(
        data_root='Dataset',
        batch_size=8,
        epochs=50,
        learning_rate=0.001,
        img_size=256
    )
