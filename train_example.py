"""
Example usage of the ImageDenoiseDataset for training a model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from creteSamples import ImageDenoiseDataset, create_data_loaders
import torchvision.transforms as transforms

# Configuration
DATA_PATH = "/media/user/f7a503ec-b25c-41a5-9baa-d350714f613a/imageData"
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


def example_training_loop():
    """Example of how to use the dataset for training"""
    
    # Optional: Define custom transforms (if needed)
    # By default, images are converted to tensors with values in [0, 1]
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any additional transforms here if needed
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        train_split=0.9,
        num_workers=4,
        transform=custom_transform
    )
    
    # Example: Simple denoising/upscaling model (replace with your actual model)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            # This is just a placeholder - replace with your actual architecture
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 3, 3, padding=1)
            self.upsample = nn.Upsample(scale_factor=1.5, mode='bicubic', align_corners=False)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            x = self.upsample(x)
            return x
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Using device: {device}")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (low_res, high_res) in enumerate(train_loader):
            # Move data to device
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(low_res)
            loss = criterion(output, high_res)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                
                output = model(low_res)
                loss = criterion(output, high_res)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Save model
    torch.save(model.state_dict(), 'denoise_model.pth')
    print("Model saved to denoise_model.pth")


def example_dataset_inspection():
    """Example of how to inspect the dataset"""
    
    # Create dataset
    dataset = ImageDenoiseDataset(
        DATA_PATH, 
        load_metadata=True  # Load metadata to see noise configurations
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        low_res, high_res, metadata = dataset[0]
        
        print(f"\nSample 0:")
        print(f"Low-res shape: {low_res.shape}")  # Should be [3, 600, 800]
        print(f"High-res shape: {high_res.shape}")  # Should be [3, 900, 1200]
        print(f"Noise config: {metadata.get('noise_config', {})}")
        
        # Visualize (optional)
        try:
            import matplotlib.pyplot as plt
            
            # Convert tensors to numpy for visualization
            low_res_np = low_res.permute(1, 2, 0).numpy()
            high_res_np = high_res.permute(1, 2, 0).numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(low_res_np)
            axes[0].set_title("Low-res (Noisy)")
            axes[0].axis('off')
            
            axes[1].imshow(high_res_np)
            axes[1].set_title("High-res (Clean)")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig('sample_pair.png')
            print("\nSample visualization saved to sample_pair.png")
        except ImportError:
            print("\nInstall matplotlib to visualize samples: pip install matplotlib")


if __name__ == "__main__":
    # Choose which example to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        example_dataset_inspection()
    else:
        example_training_loop()
