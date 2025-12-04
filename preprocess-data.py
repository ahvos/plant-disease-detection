# ===== import libraries =====
import os
from tqdm import tdqm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split


# ===== setup file paths =====
def setup_paths(data_dir):
    #create filepath and labels list
    filepaths = []
    labels = []

    #retrieve list containing names of all files/directories
    folders = os.listdir(data_dir)

    #loop through list
    for folder in folders:
        #set path of folder and list of files
        folder_path = os.path.join(data_dir, folder)
        file_list = os.listdir(folder_path)

        #loop through list of files
        for file in file_list:

            #add to filepaths and labels list
            fpath = os.path.join(folder_path, file)
            filepaths.append(fpath)
            labels.append(folder)

    #return filepaths and labels list
    return filepaths, labels


# ===== preprocess data =====



# ===== create custom dataset =====
class PlantDataset(Dataset):
    """
    reads image filenames and labels
    """