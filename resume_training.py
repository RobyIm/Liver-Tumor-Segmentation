"""
resume_training.py - Resume training from a saved checkpoint.

Loads model weights, optimizer states, and epoch counter from a checkpoint
file, then continues training with the same configuration.

Usage:
    python resume_training.py

Requires:
    - checkpoints/best_model.pth (saved by train.py)
"""

import os
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from dataset import LiverCTDataset
from models import Generator, Discriminator
from loss import L1Loss


# -------------------------
# CONFIGURATION
# -------------------------
DATA_ROOT = "/content/lits_png"
SAVE_DIR = "/content/drive/MyDrive/checkpoints"
BATCH_SIZE = 10
NUM_EPOCHS = 1000
LR_GENERATOR = 0.0002
LR_DISCRIMINATOR = 0.0002
VAL_RATIO = 0.15
IMAGE_CHANNELS = 1
NUM_DISCRIMINATOR_LAYERS = 3
NUM_WORKERS = 2
MAX_VAL_SAMPLES = 100
LOG_INTERVAL = 50
VIS_INTERVAL = 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # DATASET + TRAIN/VAL SPLIT
    # -------------------------
    full_dataset = LiverCTDataset(DATA_ROOT)

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # -------------------------
    # MODELS
    # -------------------------
    generator = Generator(input_channels=IMAGE_CHANNELS).to(device)
    discriminator = Discriminator(input_channels=IMAGE_CHANNELS).to(device)

    generator = torch.compile(generator)
    discriminator = torch.compile(discriminator)

    # -------------------------
    # LOSS & OPTIMIZERS
    # -------------------------
    criterion = L1Loss(
        critic_network=discriminator,
        num_layers=NUM_DISCRIMINATOR_LAYERS,
        num_training_images=BATCH_SIZE
    )

    generator_optimizer = optim.Adam(
        generator.parameters(), lr=LR_GENERATOR, betas=(0.5, 0.999)
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=LR_DISCRIMINATOR, betas=(0.5, 0.999)
    )

    scaler = GradScaler()

    # -------------------------
    # LOAD CHECKPOINT
    # -------------------------
    checkpoint_path = os.path.join(SAVE_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    train_losses = []
    val_losses = []

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"with Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"Resuming from epoch {start_epoch}")

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (real_masks, real_images) in enumerate(train_loader):

            real_masks = real_masks.to(device, non_blocking=True)
            real_images = real_images.to(device, non_blocking=True)

            # ----- Train Discriminator -----
            discriminator_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                fake_masks = generator(real_images)

                feat_real = discriminator.forward_all_layers(real_images * real_masks)
                feat_fake = discriminator.forward_all_layers(real_images * fake_masks.detach())

                d_loss = 0.0
                for fr, ff in zip(feat_real, feat_fake):
                    d_loss += (torch.flatten(fr, 1) - torch.flatten(ff, 1)).abs().mean()
                d_loss = d_loss / NUM_DISCRIMINATOR_LAYERS

            scaler.scale(d_loss).backward()
            scaler.step(discriminator_optimizer)
            scaler.update()

            # ----- Train Generator -----
            for p in discriminator.parameters():
                p.requires_grad = False

            generator_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                generated_masks = generator(real_images)
                g_adv_loss = criterion(real_images, generated_masks, real_masks)
                g_pixel_loss = (generated_masks - real_masks).abs().mean()
                g_loss = g_pixel_loss + 0.01 * g_adv_loss

            scaler.scale(g_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            for p in discriminator.parameters():
                p.requires_grad = True

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            if batch_idx % LOG_INTERVAL == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"D Loss: {d_loss.item():.4f} "
                    f"G Loss: {g_loss.item():.4f}"
                )

        # -------------------------
        # VALIDATION
        # -------------------------
        avg_train_loss = epoch_g_loss / len(train_loader)
        avg_val_loss = validate(generator, val_loader, criterion, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Train G Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"Best model saved at epoch {epoch} with Val Loss: {avg_val_loss:.4f}")

        # -------------------------
        # SAVE VALIDATION PREDICTIONS
        # -------------------------
        if epoch % VIS_INTERVAL == 0:
            save_val_predictions(generator, val_loader, epoch, device)
            save_loss_curves(train_losses, val_losses)


def validate(generator, val_loader, criterion, device):
    """Run validation and return average loss."""
    generator.eval()
    val_loss = 0.0

    with torch.no_grad():
        for real_masks, real_images in val_loader:
            real_masks = real_masks.to(device, non_blocking=True)
            real_images = real_images.to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                generated_masks = generator(real_images)
                loss = criterion(real_images, generated_masks, real_masks)

            val_loss += loss.item()

    generator.train()
    return val_loss / len(val_loader)


def save_val_predictions(generator, val_loader, epoch, device):
    """Save validation prediction images to disk."""
    vis_dir = os.path.join(SAVE_DIR, "val_predictions", f"epoch_{epoch:04d}")
    os.makedirs(vis_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        sample_idx = 0
        for val_masks, val_images in val_loader:
            val_images = val_images.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                val_preds = generator(val_images)

            for i in range(val_images.size(0)):
                x = val_images[i].cpu().squeeze().numpy()
                y = val_masks[i].cpu().squeeze().numpy()
                z = val_preds[i].cpu().squeeze().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].set_title("CT")
                axes[0].imshow(x, cmap="gray")
                axes[0].axis("off")
                axes[1].set_title("GT Mask")
                axes[1].imshow(y, cmap="gray")
                axes[1].axis("off")
                axes[2].set_title("Pred Mask")
                axes[2].imshow(z, cmap="gray")
                axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(vis_dir, f"sample_{sample_idx:04d}.png"),
                    dpi=100, bbox_inches='tight'
                )
                plt.close('all')

                sample_idx += 1
                if sample_idx >= MAX_VAL_SAMPLES:
                    break
            if sample_idx >= MAX_VAL_SAMPLES:
                break

    generator.train()
    print(f"Saved {sample_idx} validation predictions to {vis_dir}")


def save_loss_curves(train_losses, val_losses):
    """Save training vs validation loss plot."""
    if len(train_losses) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('G Loss')
    ax.legend()
    ax.set_title('Training vs Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=100)
    plt.close('all')


if __name__ == "__main__":
    main()
