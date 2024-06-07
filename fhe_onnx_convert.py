import torch
import torch.nn as nn
import torch.onnx as onnx
from plain_models import MLP_Bank
from tools import load_data

# convert credit and bank model to onnx for helayer

class MLP_Credit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(23, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def sigmoid(self, x):
        return 0.5 + 0.197 * x - 0.004 * (x**3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    

def convert_model(data_name):
    data_name = data_name.lower()

    if data_name == "credit":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = MLP_Credit()
    elif data_name == "bank":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = MLP_Bank()
    else:
        raise NotImplementedError(data_name)
    plain_model.load_state_dict(torch.load(f'./pretrained/{data_name}_plain.pt'))
    plain_model.eval()

    input_shape = [1] + list(x_train[0].shape)
    dummy_input = torch.rand(input_shape)
    onnx.export(plain_model, dummy_input, f'./pretrained/{data_name}_plain.onnx', 
                input_names=['inputs'], output_names=['outputs'])


if __name__ == "__main__":
    convert_model("credit")
    convert_model("bank")
    