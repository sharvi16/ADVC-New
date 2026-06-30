"""
One-time fine-tuning of DeiT-S classification head on CIFAR-10/100.
Run this BEFORE any pipeline experiments on CIFAR datasets.
Produces checkpoints/finetuned/deit_small_cifar10_head.pt (or cifar100).
"""
import os
import sys
import torch
import timm
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def finetune(dataset_name="cifar10", epochs=15, lr=1e-4, head_epochs=3):
    num_classes = 10 if dataset_name == "cifar10" else 100
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100

    # Data augmentation for training
    train_transform = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Standard transformation for validation
    val_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = DatasetClass(root="data/cifar", train=True, download=True, transform=train_transform)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    val_ds = DatasetClass(root="data/cifar", train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    print(f"[finetune] Loading pretrained DeiT-S model...")
    model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=num_classes).cuda()

    # Freeze backbone initially
    print(f"[finetune] Freezing backbone initially. Training only the classifier head for {head_epochs} epochs.")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    ckpt_dir = "checkpoints/finetuned"
    ckpt_path = os.path.join(ckpt_dir, f"deit_small_{dataset_name}_head.pt")
    
    best_acc = 0.0
    backbone_frozen = True

    for epoch in range(epochs):
        # Check if we should unfreeze the backbone
        if backbone_frozen and epoch >= head_epochs:
            print(f"[finetune] Unfreezing backbone at epoch {epoch+1} for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Re-initialize optimizer and scheduler with all parameters
            # Use slightly lower learning rate (lr / 2) for backbone fine-tuning to preserve features
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr / 2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - head_epochs)
            backbone_frozen = False

        # Train loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += images.size(0)

        epoch_val_acc = val_correct / val_total
        
        # Step the scheduler
        scheduler.step()

        # Get current learning rate
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {100 * epoch_train_acc:.2f}% | "
              f"Val Acc: {100 * epoch_val_acc:.2f}% | "
              f"LR: {curr_lr:.2e}")

        # Save the best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f" => Saved new best model with Val Acc: {100 * best_acc:.2f}%")

    print(f"\n[finetune] Fine-tuning finished! Best validation accuracy: {100 * best_acc:.2f}%")
    print(f"[finetune] Best checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    finetune(dataset_name=dataset)
