import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import numpy as np
from PIL import Image, PngImagePlugin
import torchvision.transforms as transforms
import torchvision.models as models
from creteSamples import ImageDenoiseDataset, SAVE_PATH, IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES, IMAGE_WIDTH_HIGH_RES, IMAGE_HEIGHT_HIGH_RES

PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
NUM_WORKERS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOSS_WEIGHTS = {
    "vgg": 1.2,
    "fft": 0.075,
    "l1": 1.0,
    "mae": 0.2,
    "ssim": 0.6
}

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


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights='DEFAULT').features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.layers = {
            '3': vgg[3],
            '8': vgg[8],
            '17': vgg[17],
            '26': vgg[26],
            '35': vgg[35]
        }
        self.vgg_features = vgg[:36]
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    
    def forward(self, pred, target):
        pred_normalized = (pred - self.mean) / self.std
        target_normalized = (target - self.mean) / self.std
        
        pred_features = self.vgg_features(pred_normalized)
        target_features = self.vgg_features(target_normalized)
        loss = F.mse_loss(pred_features, target_features)
        return loss


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
    
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)


class CombinedLoss(nn.Module):
    def __init__(self, device, weights=None):
        super(CombinedLoss, self).__init__()
        self.vgg_loss = VGGPerceptualLoss(device)
        self.fft_loss = FFTLoss()
        self.ssim_loss = SSIMLoss()
        self.l1_loss = nn.L1Loss()
        self.mae_loss = nn.L1Loss()
        
        if weights is None:
            weights = LOSS_WEIGHTS
        
        self.weight_vgg = weights["vgg"]
        self.weight_fft = weights["fft"]
        self.weight_l1 = weights["l1"]
        self.weight_mae = weights["mae"]
        self.weight_ssim = weights["ssim"]
    
    def forward(self, pred, target):
        vgg = self.vgg_loss(pred, target)
        fft = self.fft_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        mae = self.mae_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        total = (self.weight_vgg * vgg + 
                self.weight_fft * fft + 
                self.weight_l1 * l1 + 
                self.weight_mae * mae +
                self.weight_ssim * ssim)
        
        return total, {
            'vgg_loss': vgg.item(),
            'fft_loss': fft.item(),
            'l1_loss': l1.item(),
            'mae_loss': mae.item(),
            'ssim_loss': ssim.item(),
            'total_loss': total.item()
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, best_batch_loss=float('inf')):
    model.train()
    running_loss = 0.0
    running_losses = {'vgg': 0.0, 'fft': 0.0, 'l1': 0.0, 'mae': 0.0, 'ssim': 0.0}
    num_batches = 0
    
    batch_checkpoint_interval = 5000
    checkpoint_running_loss = 0.0
    checkpoint_batches = 0
    
    for batch_idx, (low_res, high_res) in enumerate(train_loader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        
        optimizer.zero_grad()
        
        output = model(low_res)
        loss, loss_dict = criterion(output, high_res)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss_dict['total_loss']
        running_losses['vgg'] += loss_dict['vgg_loss']
        running_losses['fft'] += loss_dict['fft_loss']
        running_losses['l1'] += loss_dict['l1_loss']
        running_losses['mae'] += loss_dict['mae_loss']
        running_losses['ssim'] += loss_dict['ssim_loss']
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Total: {loss_dict['total_loss']:.6f} "
                  f"VGG: {loss_dict['vgg_loss']:.6f} "
                  f"FFT: {loss_dict['fft_loss']:.6f} "
                  f"L1: {loss_dict['l1_loss']:.6f} "
                  f"MAE: {loss_dict['mae_loss']:.6f} "
                  f"SSIM: {loss_dict['ssim_loss']:.6f}")
            wandb.log({
                "batch/total_loss": loss_dict['total_loss'],
                "batch/vgg_loss": loss_dict['vgg_loss'],
                "batch/fft_loss": loss_dict['fft_loss'],
                "batch/l1_loss": loss_dict['l1_loss'],
                "batch/mae_loss": loss_dict['mae_loss'],
                "batch/ssim_loss": loss_dict['ssim_loss'],
                "batch": epoch * len(train_loader) + batch_idx
            })
        
        if batch_idx % 250 == 0 and batch_idx > 0:
            log_image_previews(model, train_loader, device, f"{epoch}_batch_{batch_idx}", num_images=2)
        
        checkpoint_running_loss += loss_dict['total_loss']
        checkpoint_batches += 1
        
        if batch_idx > 0 and batch_idx % batch_checkpoint_interval == 0:
            avg_checkpoint_loss = checkpoint_running_loss / checkpoint_batches
            print(f"\n[Batch Checkpoint] Average loss over {checkpoint_batches} batches: {avg_checkpoint_loss:.6f}")
            
            if avg_checkpoint_loss < best_batch_loss:
                best_batch_loss = avg_checkpoint_loss
                save_checkpoint(
                    model, optimizer, epoch, avg_checkpoint_loss,
                    CHECKPOINT_DIR / "best_batch_checkpoint.pth",
                    batch_idx=batch_idx,
                    best_batch_loss=best_batch_loss
                )
                print(f"New best batch loss: {best_batch_loss:.6f}")
            
            checkpoint_running_loss = 0.0
            checkpoint_batches = 0
    
    avg_losses = {
        'total': running_loss / num_batches,
        'vgg': running_losses['vgg'] / num_batches,
        'fft': running_losses['fft'] / num_batches,
        'l1': running_losses['l1'] / num_batches,
        'mae': running_losses['mae'] / num_batches,
        'ssim': running_losses['ssim'] / num_batches
    }
    return avg_losses, best_batch_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_losses = {'vgg': 0.0, 'fft': 0.0, 'l1': 0.0, 'mae': 0.0, 'ssim': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            output = model(low_res)
            loss, loss_dict = criterion(output, high_res)
            
            running_loss += loss_dict['total_loss']
            running_losses['vgg'] += loss_dict['vgg_loss']
            running_losses['fft'] += loss_dict['fft_loss']
            running_losses['l1'] += loss_dict['l1_loss']
            running_losses['mae'] += loss_dict['mae_loss']
            running_losses['ssim'] += loss_dict['ssim_loss']
            num_batches += 1
    
    avg_losses = {
        'total': running_loss / num_batches,
        'vgg': running_losses['vgg'] / num_batches,
        'fft': running_losses['fft'] / num_batches,
        'l1': running_losses['l1'] / num_batches,
        'mae': running_losses['mae'] / num_batches,
        'ssim': running_losses['ssim'] / num_batches
    }
    return avg_losses


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


def save_checkpoint(model, optimizer, epoch, loss, path, batch_idx=0, best_batch_loss=None):
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_batch_loss': best_batch_loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_idx = checkpoint.get('batch_idx', 0)
    best_batch_loss = checkpoint.get('best_batch_loss', float('inf'))
    print(f"Checkpoint loaded: {path}, Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.6f}")
    return epoch, loss, batch_idx, best_batch_loss


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
            "loss_weights": LOSS_WEIGHTS,
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
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    model = DenoiseUpscaleNet(MODEL_CONFIG).to(DEVICE)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    wandb.config.update({
        "num_parameters": num_params,
        "model_config": MODEL_CONFIG
    })
    
    criterion = CombinedLoss(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    best_batch_loss = float('inf')
    start_epoch = 1
    
    batch_checkpoint_path = CHECKPOINT_DIR / "best_batch_checkpoint.pth"
    epoch_checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    checkpoint_to_load = None
    if batch_checkpoint_path.exists():
        checkpoint_to_load = batch_checkpoint_path
        print(f"Found batch checkpoint: {batch_checkpoint_path}")
    elif epoch_checkpoint_path.exists():
        checkpoint_to_load = epoch_checkpoint_path
        print(f"Found epoch checkpoint: {epoch_checkpoint_path}")
    
    if checkpoint_to_load:
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            epoch_data = load_checkpoint(model, optimizer, checkpoint_to_load)
            start_epoch = epoch_data[0] + 1
            best_val_loss = epoch_data[1]
            if len(epoch_data) >= 4:
                best_batch_loss = epoch_data[3]
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        train_losses, best_batch_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, DEVICE, best_batch_loss)
        val_losses = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train - Total: {train_losses['total']:.6f}, VGG: {train_losses['vgg']:.6f}, "
              f"FFT: {train_losses['fft']:.6f}, L1: {train_losses['l1']:.6f}, "
              f"MAE: {train_losses['mae']:.6f}, SSIM: {train_losses['ssim']:.6f}")
        print(f"Val   - Total: {val_losses['total']:.6f}, VGG: {val_losses['vgg']:.6f}, "
              f"FFT: {val_losses['fft']:.6f}, L1: {val_losses['l1']:.6f}, "
              f"MAE: {val_losses['mae']:.6f}, SSIM: {val_losses['ssim']:.6f}")
        
        wandb.log({
            "epoch": epoch,
            "train/total_loss": train_losses['total'],
            "train/vgg_loss": train_losses['vgg'],
            "train/fft_loss": train_losses['fft'],
            "train/l1_loss": train_losses['l1'],
            "train/mae_loss": train_losses['mae'],
            "train/ssim_loss": train_losses['ssim'],
            "val/total_loss": val_losses['total'],
            "val/vgg_loss": val_losses['vgg'],
            "val/fft_loss": val_losses['fft'],
            "val/l1_loss": val_losses['l1'],
            "val/mae_loss": val_losses['mae'],
            "val/ssim_loss": val_losses['ssim'],
        })
        
        log_image_previews(model, val_loader, DEVICE, epoch)
        
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, epoch, val_losses['total'],
                CHECKPOINT_DIR / "best_model.pth",
                batch_idx=0,
                best_batch_loss=best_batch_loss
            )
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch
        
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_losses['total'],
                CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth",
                batch_idx=0,
                best_batch_loss=best_batch_loss
            )
    
    print("\nTraining completed!")
    wandb.finish()


if __name__ == "__main__":
    main()