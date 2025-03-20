import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
import wandb

wandb.init(project="tumor-segmentation", name="test_org")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class TumorDataset(Dataset):
    def __init__(self, image_folder, mask_folder, add_noise=False):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.add_noise = add_noise
        self.file_names = os.listdir(image_folder)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.file_names[idx])
        mask_path = os.path.join(self.mask_folder, self.file_names[idx])
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)
        image = torch.tensor(image).unsqueeze(0) 
        mask = torch.tensor(mask).unsqueeze(0)
        if self.add_noise:
            noise = torch.randn_like(image) * 25  
            image = torch.clamp(image + noise, 0, 255)
        return image, mask

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        smooth = 1e-5
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature*2, feature))
            
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) 
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  
        return torch.sigmoid(self.final_conv(x))

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    print(f"Saving checkpoint at epoch {epoch}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth", weights_only=False):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device) 
        model.load_state_dict(checkpoint['model_state_dict'])
        if not weights_only and optimizer is not None:  
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from epoch {epoch}, loss {loss:.4f}")
        return model, optimizer, epoch, loss
    else:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0, None

def log_to_wandb(images, masks, preds):
    slice_idx = images.shape[2] // 2
    wandb.log({
        "image": wandb.Image(images[0, 0, slice_idx].detach().cpu().numpy(), caption="Input Image"),
        "mask": wandb.Image(masks[0, 0, slice_idx].detach().cpu().numpy(), caption="Ground Truth Mask"),
        "prediction": wandb.Image(preds[0, 0, slice_idx].detach().cpu().numpy(), caption="Predicted Mask")
    })

def train_model(model, dataloader, criterion, optimizer, epochs=5, checkpoint_filename="checkpoint.pth"):
    model.to(device)
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_filename)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for images, masks in tqdm(dataloader, total=len(dataloader)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            log_to_wandb(images, masks, outputs)

        avg_loss = total_loss / len(dataloader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
        save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_filename)

def evaluate_model(model, dataloader, checkpoint_filename="checkpoint.pth"):
    model.eval()
    model, _, _, _ = load_checkpoint(model, None, checkpoint_filename, weights_only=True)
    model.to(device)

    dice_scores = []
    iou_scores = []
    accuracy_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            log_to_wandb(images, masks, preds)

            dice_scores.append(1 - DiceLoss()(preds, masks).item())
            iou_scores.append(jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten()))
            
            accuracy = accuracy_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            accuracy_scores.append(accuracy)

    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}, Mean IoU: {np.mean(iou_scores):.4f}, Mean Accuracy: {np.mean(accuracy_scores):.4f}")


image_folder = "/home/woody/iwi5/iwi5207h/case_study/data/extracted/img"
mask_folder = "/home/woody/iwi5/iwi5207h/case_study/data/extracted/seg"

dataset = TumorDataset(image_folder, mask_folder, add_noise=False)
dataset_noisy = TumorDataset(image_folder, mask_folder, add_noise=True)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset_noisy, test_dataset_noisy = random_split(dataset_noisy, [train_size, test_size])

dataloader_train = DataLoader(train_dataset, batch_size=4, shuffle=True)  
dataloader_train_noisy = DataLoader(train_dataset_noisy, batch_size=4, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)  
dataloader_test_noisy = DataLoader(test_dataset_noisy, batch_size=1, shuffle=False)

model = UNet3D()
criterion = DiceLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.005)

train_model(model, dataloader_train_noisy, criterion, optimizer, epochs=8)
evaluate_model(model, dataloader_test)  
evaluate_model(model, dataloader_test_noisy)  