"""
One-time fine-tuning of DeiT-S classification head on CIFAR-10/100.
Run this BEFORE any pipeline experiments on CIFAR datasets.
Produces checkpoints/finetuned/deit_small_cifar10_head.pt (or cifar100).
"""
import torch, timm
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def finetune(dataset_name="cifar10", epochs=5, lr=1e-4):
    num_classes = 10 if dataset_name == "cifar10" else 100
    DatasetClass = CIFAR10 if dataset_name == "cifar10" else CIFAR100

    transform = T.Compose([
        T.Resize(224),  # upsample 32x32 -> 224x224
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_ds = DatasetClass(root="data/cifar", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} done")

    import os
    os.makedirs("checkpoints/finetuned", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/finetuned/deit_small_{dataset_name}_head.pt")

if __name__ == "__main__":
    import sys
    finetune(sys.argv[1] if len(sys.argv) > 1 else "cifar10")
