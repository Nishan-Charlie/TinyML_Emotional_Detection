import os
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model
from LightCBAMNet import LightMixerWithCBAM

def get_args():
    parser = argparse.ArgumentParser(description='Train LightCBAMNet')
    parser.add_argument('--data_dir', type=str, default='images', help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    args = get_args()
    set_seed(args.seed)
    print(f"Using device: {args.device}")
    
    # Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Data Loaders
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        print(f"Error: Data directories not found at {train_dir} or {valid_dir}")
        return

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Model Setup
    model = LightMixerWithCBAM(num_classes=num_classes)
    model = model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 10)
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, args.device)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        
        val_loss, val_acc = validate(model, dataloaders['valid'], criterion, args.device)
        print(f'Valid Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_lightcbamnet.pth')
            print(f"New best model saved with Acc: {val_acc:.2f}%")
            
        print()
        
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}%')

if __name__ == '__main__':
    main()
