
import torch
import pandas as pd
from torch.utils.data import random_split, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def encode_numeric_zscore(x, mean=None, sd=None):
    if mean is None:
        mean = x.mean()

    if sd is None:
        sd = x.std()

    return (x - mean) / sd

def get_dataset(x_csv_path: str, y_csv_path: str, test_size: float = 0.2, random_state: int = 42):

    x = pd.read_csv("Active_Wiretap_dataset.csv")  
    y = pd.read_csv("Active_Wiretap_labels.csv")  

    
    for column in x.columns:
        if pd.api.types.is_numeric_dtype(x[column]):
            x[column] = encode_numeric_zscore(x[column])

    imputer = SimpleImputer(strategy='mean')
    x = imputer.fit_transform(x)
    encoder = OneHotEncoder()

    y = y.values
    labels = y[:, 1]
    labels = labels[:-1]
    labels = labels.reshape(-1, 1)
    labels = encoder.fit_transform(labels)
    labels = labels.toarray()

    
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=test_size, random_state=random_state)

    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = y_train.astype(np.float32)  
    y_train = torch.from_numpy(y_train)
    y_test = y_test.astype(np.float32)
    y_test = torch.from_numpy(y_test)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset,test_dataset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_dataset("Active_Wiretap_dataset.csv","Active_Wiretap_labels.csv",0.2,42)
    total_samples = len(trainset)
    num_samples_per_partition = total_samples // num_partitions
    remainder = total_samples % num_partitions

    partition_sizes = [num_samples_per_partition] * num_partitions

    for i in range(remainder):
        partition_sizes[i] += 1

    trainloaders = []
    valloaders = []

    start_idx = 0

    for size in partition_sizes:
        end_idx = start_idx + size
        partition, _ = random_split(trainset, [size, total_samples - size])
        num_total = len(partition)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        trainset_, valset = random_split(partition, [num_train, num_val])

        trainloaders.append(
            DataLoader(trainset_, batch_size=batch_size, shuffle=True, num_workers=2)
        )

        valloaders.append(
            DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
        )

        start_idx = end_idx

    testloader = DataLoader(testset, batch_size=2)

    return trainloaders, valloaders, testloader






    





