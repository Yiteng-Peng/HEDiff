import torch
import tenseal as ts


class MNISTCryptoNet_TS:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x.square_()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x.square_()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



class DigitsCryptoNet_TS:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # sigmoid activation
        enc_x = enc_x.polyval([0.5, 0.197, 0, -0.004])
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval([0.5, 0.197, 0, -0.004])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class CreditMLP_TS:
    def __init__(self, torch_nn):

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()


    def forward(self, enc_x):
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval([0.5, 0.197, 0, -0.004])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class BankMLP_TS:
    def __init__(self, torch_nn):

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()


    def forward(self, enc_x):
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval([0, 1, 1])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def PredictPlainVector(plain_model, data):
    pred_p = plain_model(data)
    label_p = pred_p.argmax(1)
    return pred_p, label_p


def PredictEncVector(enc_model, data, context):
    x_enc = ts.ckks_vector(context, data[0])
    enc_output = enc_model(x_enc)
    output = enc_output.decrypt()
    pred_e = torch.tensor(output).view(1, -1)
    label_e = pred_e.argmax(1)
    return pred_e, label_e


def PredictConvEncVector(enc_model, data, context, kernel_shape, stride):
    x_enc, windows_nb = ts.im2col_encoding(
        context, data.view(data.shape[-2], data.shape[-1]).tolist(), kernel_shape[0],
        kernel_shape[1], stride
    )
    enc_output = enc_model(x_enc, windows_nb)
    output = enc_output.decrypt()
    pred_e = torch.tensor(output).view(1, -1)
    label_e = pred_e.argmax(1)
    return pred_e, label_e