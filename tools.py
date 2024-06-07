import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


TEST_SIZE = 0.2


def load_features_num(data_name):
    if data_name == "digits":
        in_features = 8 * 8
        out_features = 10
    elif data_name == "bank":
        in_features = 20
        out_features = 2
    elif data_name == "credit":
        in_features = 23
        out_features = 2
    elif data_name == "car":
        in_features = 6
        out_features = 4
    elif data_name == "mnist":
        in_features = 28 * 28 * 1
        out_features = 10
    elif data_name == "cifar":
        in_features = 32 * 32 * 3
        out_features = 10
    else:
        raise NotImplementedError(data_name)

    return in_features, out_features


def load_features_label(data_name):
    if data_name == "digits":
        features = np.load('./dataset/digits_data.npy')
        labels = np.load('./dataset/digits_label.npy')
    elif data_name == "bank":
        features = np.load('./dataset/bank_under_sampling_data.npy')
        labels = np.load('./dataset/bank_under_sampling_label.npy')
    elif data_name == "credit":
        features = np.load('./dataset/credit_under_sampling_data.npy')
        labels = np.load('./dataset/credit_under_sampling_label.npy')
    elif data_name == "car":
        features = np.load('./dataset/car_data.npy')
        labels = np.load('./dataset/car_label.npy')
    else:
        raise NotImplementedError(data_name)

    return features, labels


def load_torch_data(data_name, example=False, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if data_name == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./dataset', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=False)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./dataset', train=False, transform=transform),
            batch_size=batch_size, shuffle=False)
    elif data_name == "cifar":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./dataset', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=False)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./dataset', train=False, transform=transform),
            batch_size=batch_size, shuffle=False)

    if example:
        sample_list = []
        sample_num = 10000
        for data, label in train_loader:
            if sample_num == 0:
                break
            else:
                sample_num -= 1
            sample_list.append(data)
        return train_loader, test_loader, torch.cat(sample_list).numpy()
    else:
        return train_loader, test_loader


def load_train_data(data_name, train_num=None):
    features, labels = load_features_label(data_name)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE, shuffle=False)
    if train_num is not None:
        train_num = len(x_train) if train_num > len(x_train) else train_num
        _, x_train, _, y_train = train_test_split(x_train, y_train, test_size=train_num, shuffle=True)

    in_features, out_features = load_features_num(data_name)

    return torch.Tensor(x_train), torch.Tensor(y_train), in_features, out_features


def load_data(data_name, example=False, batch_size=64):
    features, labels = load_features_label(data_name)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE, shuffle=False)
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # in_features, out_features = load_features_num(data_name)

    if example:
        return train_loader, test_loader, x_train
    else:
        return train_loader, test_loader
    

def load_tf_data(data_name, train_num=None):
    if data_name == "digits":
        features, labels = load_features_label(data_name)
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE, shuffle=False)
        img_rows, img_cols = 8, 8
    elif data_name == "mnist":
        train_loader, test_loader = load_torch_data(data_name)
        x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []
        for data, label in train_loader:
            x_train_list.append(data)
            y_train_list.append(label)
        for data, label in test_loader:
            x_test_list.append(data)
            y_test_list.append(label)
            
        x_train = torch.cat(x_train_list).numpy()
        x_test = torch.cat(x_test_list).numpy()
        y_train = torch.cat(y_train_list).numpy()
        y_test = torch.cat(y_test_list).numpy()
        img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    return x_train, x_test, y_train, y_test
