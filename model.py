# ===== library imports =====
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

# ===== code imports =====
from preprocess_data import preprocess_data

# ===== setup device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== setup model function =====
def setup_model(num_classes):
    """
    load model and replace final FC layer to match number of classes in dataset.
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
    """

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
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

        #save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        #print epoch progress
        print(f"epoch [{epoch+1}/{epochs}] | "
            f"train loss: {avg_train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {avg_val_loss:.4f} | "
            f"val acc: {val_accuracy:.4f}")
        
    return train_losses, val_losses, train_accuracies, val_accuracies



# ===== testing set evaluation function =====
def evaluate_on_test(model, test_loader, loss_fn):
    """
    evaluate test data
    """
    model.eval()
    test_loss_total, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            test_loss_total += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            test_correct += preds.eq(labels).sum().item()
            test_total += labels.size(0)

    avg_test_loss = test_loss_total / test_total
    test_accuracy = test_correct / test_total

    print(f"\n[TESTING SET] loss: {avg_test_loss:.4f} | acc: {test_accuracy:.4f}")
    return avg_test_loss, test_accuracy





# ===== MAIN FUNCTION =====
def main():
    print(f"using device: {device}")

    #set data directory path
    data_dir = r"datasets\plant_leave_diseases_dataset_without_augmentation"

    #hyperparameters
    batch_size = 64
    learning_rate = 0.0001
    epochs = 25

    #preprocess data
    train_loader, val_loader, test_loader, le, num_classes = preprocess_data(
        data_dir=data_dir,
        img_size=224,
        batch_size=batch_size,
        subset_size=5000,
    )

    #setup model
    model = setup_model(num_classes=num_classes)

    #loss and optimizer definitions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #train data
    print("training data...")
    train_losses, val_losses, train_acc, val_acc = train_data(
        model=model,
        epochs=epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    epochs_range = range(1, epochs + 1)

    # plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label="train loss", marker='o')
    plt.plot(epochs_range, val_losses, label="validation loss", marker='o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training and validation loss curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_acc, label="train accuracy", marker='o')
    plt.plot(epochs_range, val_acc, label="validation accuracy", marker='o')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("training and validation accuracy curve")
    plt.legend()
    plt.grid(True)
    plt.show()


    #evaluate on test set (used after final settings chosen)
    print("evaluating on test set...")
    evaluate_on_test(model, test_loader, loss_fn)


if __name__ == "__main__":
    main()