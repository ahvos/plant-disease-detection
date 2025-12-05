# ===== library imports =====
from tqdm import tdqm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split

# ===== code imports =====
from preprocess_data import preprocess_data


# ===== setup model function =====
def setup_model():
    #load resnet18 model
    model = models.resnet18(pretrained=True)

    #replace final connected later with 2 outputs
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)


# ===== train data function =====
def train_data(model, epochs, optimizer, loss_fn, train_loader, validation_loader):
    for epoch, in range(epochs):
        model.train()
        train_loss_total, train_correct, train_total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()               #reset gradients
            outputs = model(imgs)               #forward pass
            loss = loss_fn(outputs, labels)     #compute loss
            loss.backward()                     #backpropagation
            optimizer.step()                    #update weights

            #track accuracy and loss
            train_loss_total += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        
        #calculate accuracy and average loss
        avg_train_loss = train_loss_total / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss_total, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in validation_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                val_loss_total += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total

        #print epoch progress
        print(f"epoch [{epoch+1}/{epochs}] | "
            f"train loss: {avg_train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {avg_val_loss:.4f} | "
            f"val acc: {val_accuracy:.4f}")



# ===== MAIN FUNCTION =====
def main():
    #preprocess data
    data_dir = "..\datasets\plant_leave_diseases_dataset_without_augmentation"

    #create datasets
    

    #create dataloader

    #setup model

    #build model

    #definitions
    batch_size = 64
    learning_rate = 0.001
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    #train data


if __name__ == "__main__":
    main()