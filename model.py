# ===== library imports =====
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# ===== code imports =====
from preprocess_data import preprocess_data

# ===== setup device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


# ===== setup model function =====
def setup_model(num_classes):
    """
    load model and replace final FC layer to match number of 
    classes in dataset.
    """

    #load resnet18 model
    model = models.resnet18(pretrained=True)

    #replace final connected later with 2 outputs
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model


# ===== train data function =====
def train_data(model, epochs, optimizer, loss_fn, train_loader, validation_loader):
    """
    training loop with validation at each epoch
    
    :param model: Description
    :param epochs: Description
    :param optimizer: Description
    :param loss_fn: Description
    :param train_loader: Description
    :param validation_loader: Description
    """

    for epoch, in range(epochs):
        model.train()
        train_loss_total, train_correct, train_total = 0.0, 0, 0

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

        #validation
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
    #set data directory path
    data_dir = r"datasets\plant_leave_diseases_dataset_without_augmentation"

    #hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    #preprocess data
    train_loader, val_loader, test_loader, le, num_classes = preprocess_data(
        data_dir=data_dir,
        img_size=224,
        batch_size=batch_size,
        subset_size=200,
    )

    #setup model
    model = setup_model(num_classes=num_classes)

    #loss and optimizer definitions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #train data
    train_data(
        model=model,
        epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=val_loader,
    )


if __name__ == "__main__":
    main()