import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from tools import load_data, load_torch_data


class CryptoNet_MNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, stride=3, padding=0)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(1)
        x = x * x
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


class CryptoNet_Digits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(180, 64)
        self.fc2 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(1)
        x = self.sigmoid(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    

class CryptoNet_MNIST_helayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, stride=3, padding=0)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.flatten(1)
        x = x * x
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


class CryptoNet_Digits_helayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(180, 64)
        self.fc2 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.flatten(1)
        x = self.sigmoid(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class MLP_Credit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(23, 128)
        self.fc2 = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class MLP_Bank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = x * x + x
        x = self.fc2(x)
        return x


def train(model, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def tf2torch():
    pass


def Credit_plain_train():
    model = MLP_Credit()

    epochs = 100
    train_loader, test_loader = load_data("credit")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train(model, train_loader, optimizer, criterion)
        acc = test(model, train_loader)
        tacc = test(model, test_loader)
        pbar.set_postfix({'Epoch': epoch, 'Accuracy': f'{acc:.2f}%,{tacc:.2f}%'})
    torch.save(model.state_dict(), f'./pretrained/credit_plain.pt')


def Bank_plain_train():
    model = MLP_Bank()

    epochs = 100
    train_loader, test_loader = load_data("bank")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)
        pbar.set_postfix({'Epoch': epoch, 'Accuracy': f'{acc:.2f}%'})
    torch.save(model.state_dict(), f'./pretrained/bank_plain.pt')


def MNIST_plain_train():
    model = CryptoNet_MNIST()

    epochs = 100
    train_loader, test_loader = load_torch_data("mnist")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)
        pbar.set_postfix({'Epoch': epoch, 'Accuracy': f'{acc:.2f}%'})
    torch.save(model.state_dict(), f'./pretrained/mnist_plain.pt')


def Digits_plain_train():
    model = CryptoNet_Digits()

    epochs = 100
    train_loader, test_loader = load_data("digits")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)
        pbar.set_postfix({'Epoch': epoch, 'Accuracy': f'{acc:.2f}%'})
    torch.save(model.state_dict(), f'./pretrained/digits_plain.pt')


def Credit_plain_test():
    model = MLP_Credit()
    model.load_state_dict(torch.load(f'./pretrained/credit_plain.pt'))
    train_loader, test_loader = load_data("credit")
    acc = test(model, test_loader)
    print(f"credit: {acc:.2f}%")


def Bank_plain_test():
    model = MLP_Bank()
    model.load_state_dict(torch.load(f'./pretrained/bank_plain.pt'))
    train_loader, test_loader = load_data("bank")
    acc = test(model, test_loader)
    print(f"bank: {acc:.2f}%")


def MNIST_plain_test():
    model = CryptoNet_MNIST()
    model.load_state_dict(torch.load(f'./pretrained/mnist_plain.pt'))
    train_loader, test_loader = load_torch_data("mnist")
    acc = test(model, test_loader)
    print(f"mnist: {acc:.2f}%")


def Digits_plain_test():
    model = CryptoNet_Digits()
    model.load_state_dict(torch.load(f'./pretrained/digits_plain.pt'))
    train_loader, test_loader = load_data("digits")
    acc = test(model, test_loader)
    print(f"digits: {acc:.2f}%")


def MNIST_tf_plain_test():
    model = CryptoNet_MNIST_helayers()
    model.load_state_dict(torch.load(f'./pretrained/mnist_plain_tf.pt'))
    train_loader, test_loader = load_torch_data("mnist")
    acc = test(model, test_loader)
    print(f"mnist: {acc:.2f}%")


def Digits_tf_plain_test():
    model = CryptoNet_Digits_helayers()
    model.load_state_dict(torch.load(f'./pretrained/digits_plain_tf.pt'))
    train_loader, test_loader = load_data("digits")
    acc = test(model, test_loader)
    print(f"digits: {acc:.2f}%")


if __name__ == "__main__":
    Credit_plain_train()
    Bank_plain_train()
    MNIST_plain_train()
    Digits_plain_train()

    Credit_plain_test()
    Bank_plain_test()
    MNIST_plain_test()
    Digits_plain_test()

    MNIST_tf_plain_test()
    Digits_tf_plain_test()

    pass