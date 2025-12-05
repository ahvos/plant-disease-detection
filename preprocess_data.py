# ===== import libraries =====
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ===== setup file paths =====
def setup_paths(data_dir):
    """
    walks through each class folder and collect the image paths and labels.
    
    :param data_dir: Description 
    """
    #create filepath and labels list
    filepaths = []
    labels = []

    #retrieve list containing names of all class folders
    folders = sorted(os.listdir(data_dir))

    #loop through list
    for folder in folders:
        #set path of folder and list of files
        folder_path = os.path.join(data_dir, folder)
        file_list = os.listdir(folder_path)

        #skip non-folders
        if not os.path.isdir(folder_path):
            continue

        #loop through list of files
        for file in file_list:
            #add to filepaths and labels list
            fpath = os.path.join(folder_path, file)
            
            #ensure file is correct image type
            if fpath.lower().endswith((",jpg", ".jpeg", ".png")):
                filepaths.append(fpath)
                labels.append(folder)

    #return filepaths and labels list
    return filepaths, labels



# ===== create custom dataset =====
class PlantDataset(Dataset):
    """
    dataset that loads images filenames and integer-encoded labels.
    """

    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        img_path = self.filepaths(index)
        label = self.labels[index]

        #load image
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label



# ===== data transforms and loaders function =====
def transform_data(img_size=224):
    """
    returns train and validation/test transfroms.

    :param img_size: set size of image
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size, img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    validation_transform = transforms.Compose([
        transforms.Resize(img_size, img_size), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, validation_transform



# ===== preprocess data function =====
def preprocess_data(data_dir, img_size=224, batch_size=64, subset_size: int | None = None, val_split=0.15, test_split=0.15):
    """
    preprocess_data pipeline:
        1. collect image paths and string labels
        2. encode labels with LabelEncoder
        3. split data into train / val / test
        4. apply transformers
        5. build dataloaders
    
    returns: train_loader, val_loader, test_loader, label_encoded, num_classes
    """

    #load in filepaths and labels
    print("loading file paths and labels...")
    filepaths, labels = setup_paths(data_dir)


    #limit dataset size for quick tests
    if subset_size is not None:
        filepaths = filepaths[:subset_size]
        labels_str = labels_str[:subset_size]
        print(f"running in SUBSET MODE: {subset_size} images")

    #encode string labels to ints
    print("encoding labels...")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)

    #total validation and test percentage
    val_test_total = val_split + test_split

    #split training and validation/test dataset
    print("splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths,
        labels,
        test_size=val_test_total,
        stratify=labels,
        random_state=13
    )

    #split validation and test dataset
    val_ratio_total = val_split / (val_split + test_split)
    
    print("splitting validation and test dataset...")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1-val_ratio_total),
        stratify=y_temp,
        random_state=13,
    )

    #display split info
    print(f"train size: {len(X_train)}, val size: {len(X_val)}, test size: {len(X_test)}")


    #get transforms
    train_transfrom, val_transform = transform_data(img_size=img_size)

    #build datasets
    train_dataset = PlantDataset(X_train, y_train, transform=train_transfrom)
    val_dataset = PlantDataset(X_val, y_val, transform=val_transform)
    test_dataset = PlantDataset(X_test, y_test, transform=val_transform)

    #build dataloaders for train
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    #build dataloaders for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    #build dataloaders for test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("DATA PREPROCESSING COMPLETE.")
    return train_loader, val_loader, test_loader, le, num_classes

if __name__ == "__main__":
    # change this to your PlantVillage root folder
    DATA_DIR = "path/to/PlantVillage"

    train_loader, val_loader, test_loader, le, num_classes = preprocess_data(
        data_dir=DATA_DIR,
        img_size=224,
        batch_size=32,
    )

    # Peek at one batch
    images, labels = next(iter(train_loader))
    print("Batch image tensor shape:", images.shape)
    print("Batch labels shape:", labels.shape)