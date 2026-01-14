import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import numpy as np
from PIL import Image, PngImagePlugin
import torchvision.transforms as transforms
from creteSamples import ImageDenoiseDataset, SAVE_PATH, IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES, IMAGE_WIDTH_HIGH_RES, IMAGE_HEIGHT_HIGH_RES

PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "encoder_channels": [64, 128, 256, 512],
    "bottleneck_channels": 1024,
    "convs_per_block": [2, 2, 3, 3],
    "decoder_convs_per_block": [3, 3, 2, 2],
}

class DenoiseUpscaleNet(nn.Module):
    def __init__(self, config):
        super(DenoiseUpscaleNet, self).__init__()
        
        self.config = config
        encoder_channels = config["encoder_channels"]
        bottleneck_channels = config["bottleneck_channels"]
        convs_per_block = config["convs_per_block"]
        decoder_convs_per_block = config["decoder_convs_per_block"]
        
        self.depth = len(encoder_channels)
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = 3
        for i, (out_channels, num_convs) in enumerate(zip(encoder_channels, convs_per_block)):
            layers = []
            for j in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            self.encoders.append(nn.Sequential(*layers))
            self.pools.append(nn.MaxPool2d(2, 2))
        
        bottleneck_layers = []
        in_channels = encoder_channels[-1]
        for _ in range(3):
            bottleneck_layers.append(nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1))
            bottleneck_layers.append(nn.ReLU(inplace=True))
            in_channels = bottleneck_channels
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        in_channels = bottleneck_channels
        for i in range(self.depth - 1, -1, -1):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
            skip_channels = encoder_channels[i]
            decoder_in_channels = in_channels + skip_channels
            decoder_out_channels = encoder_channels[i]
            num_convs = decoder_convs_per_block[i]
            
            layers = []
            for j in range(num_convs):
                layers.append(nn.Conv2d(decoder_in_channels, decoder_out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                decoder_in_channels = decoder_out_channels
            
            self.decoders.append(nn.Sequential(*layers))
            in_channels = decoder_out_channels
        
        self.upscale_to_target = nn.Upsample(
            size=(IMAGE_HEIGHT_HIGH_RES, IMAGE_WIDTH_HIGH_RES), 
            mode='bilinear', 
            align_corners=False
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0], 3, kernel_size=1),
        )
    
    def forward(self, x):
        encoder_outputs = []
        
        for i in range(self.depth):
            x = self.encoders[i](x)
            encoder_outputs.append(x)
            x = self.pools[i](x)
        
        x = self.bottleneck(x)
        
        for i in range(self.depth):
            x = self.upsamples[i](x)
            skip = encoder_outputs[self.depth - 1 - i]
            
            if x.shape[2:] != skip.shape[2:]:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                          diff_h // 2, diff_h - diff_h // 2])
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
        
        x = self.upscale_to_target(x)
        x = self.final(x)
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (low_res, high_res) in enumerate(train_loader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        
        optimizer.zero_grad()
        
        output = model(low_res)
        loss = criterion(output, high_res)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 1 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
            wandb.log({
                "batch_loss": loss.item(),
                "batch": epoch * len(train_loader) + batch_idx
            })
        
        if batch_idx % 500 == 0 and batch_idx > 0:
            log_image_previews(model, train_loader, device, f"{epoch}_batch_{batch_idx}", num_images=2)
    
    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            output = model(low_res)
            loss = criterion(output, high_res)
            
            running_loss += loss.item()
            num_batches += 1
    
    avg_loss = running_loss / num_batches
    return avg_loss


def log_image_previews(model, val_loader, device, epoch, num_images=4):
    model.eval()
    was_training = model.training
    
    with torch.no_grad():
        low_res, high_res = next(iter(val_loader))
        low_res = low_res[:num_images].to(device)
        high_res = high_res[:num_images].to(device)
        
        output = model(low_res)
        
        low_res_np = low_res.cpu().numpy().transpose(0, 2, 3, 1)
        high_res_np = high_res.cpu().numpy().transpose(0, 2, 3, 1)
        output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
        
        low_res_np = np.clip(low_res_np, 0, 1)
        high_res_np = np.clip(high_res_np, 0, 1)
        output_np = np.clip(output_np, 0, 1)
        
        images_to_log = []
        
        for i in range(num_images):
            images_to_log.append(wandb.Image(
                low_res_np[i],
                caption=f"Epoch {epoch} - Input (Noisy Low-Res)"
            ))
            images_to_log.append(wandb.Image(
                output_np[i],
                caption=f"Epoch {epoch} - Output (Denoised High-Res)"
            ))
            images_to_log.append(wandb.Image(
                high_res_np[i],
                caption=f"Epoch {epoch} - Target (Clean High-Res)"
            ))
        
        wandb.log({
            "preview_images": images_to_log,
        })
    
    if was_training:
        model.train()


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {path}, Epoch: {epoch}, Loss: {loss:.6f}")
    return epoch, loss


def main():
    wandb.init(
        project="image-denoise-upscale",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "architecture": "DenoiseUpscaleNet",
            "input_size": f"{IMAGE_WIDTH_LOW_RES}x{IMAGE_HEIGHT_LOW_RES}",
            "output_size": f"{IMAGE_WIDTH_HIGH_RES}x{IMAGE_HEIGHT_HIGH_RES}",
        }
    )
    
    print(f"Using device: {DEVICE}")
    
    transform = transforms.ToTensor()
    
    print("Loading dataset...")
    full_dataset = ImageDenoiseDataset(SAVE_PATH, transform=transform)
    
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    model = DenoiseUpscaleNet(MODEL_CONFIG).to(DEVICE)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    wandb.config.update({
        "num_parameters": num_params,
        "model_config": MODEL_CONFIG
    })
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    start_epoch = 1
    
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if checkpoint_path.exists():
        print(f"Found checkpoint: {checkpoint_path}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        
        log_image_previews(model, val_loader, DEVICE, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                CHECKPOINT_DIR / "best_model.pth"
            )
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch
        
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
            )
    
    print("\nTraining completed!")
    wandb.finish()


if __name__ == "__main__":
    main()
