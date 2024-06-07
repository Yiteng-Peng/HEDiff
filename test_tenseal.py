from plain_models import MLP_Credit, MLP_Bank, CryptoNet_Digits, CryptoNet_MNIST
import numpy as np
import logging
import pickle
import datetime
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils
from tools import load_data, load_torch_data
from tqdm import tqdm
import tenseal as ts
from torchattacks.attack import Attack
from random import sample
import argparse


from base_ts import *


log_filename = datetime.datetime.now().strftime("./log/ts_test.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_with_tenseal(pln_model, enc_model, test_loader, context, kernel_shape=None, stride=None):
    test_data_num = len(test_loader.dataset)

    all_y_pln_pred = np.zeros((test_data_num), dtype=np.int64)
    all_y_enc_pred = np.zeros((test_data_num), dtype=np.int64)
    all_targets = np.ones((test_data_num), dtype=np.int64)

    idx = 0
    pbar = tqdm(test_loader)
    for data, target in pbar:
        target = target.numpy()
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target

        y_pln_pred = pln_model(data).detach().numpy()
        y_pln_pred = np.argmax(y_pln_pred, axis=1)

        if kernel_shape is not None:
            _, y_enc_pred = PredictConvEncVector(enc_model, data, context, kernel_shape, stride)
        else:
            _, y_enc_pred = PredictEncVector(enc_model, data, context)

        all_y_pln_pred[idx:endidx] = y_pln_pred
        all_y_enc_pred[idx:endidx] = y_enc_pred

        idx += target.shape[0]

        pbar.set_postfix({'pln acc cnt': np.sum(all_targets == all_y_pln_pred), 
                          'enc acc cnt': np.sum(all_targets == all_y_enc_pred)})

    n_pln_correct = np.sum(all_targets == all_y_pln_pred)
    n_enc_correct = np.sum(all_targets == all_y_enc_pred)
    n_deviation = np.sum(np.logical_and(all_targets == all_y_pln_pred, all_y_pln_pred != all_y_enc_pred))

    pln_acc = n_pln_correct * 100 / test_data_num
    enc_acc = n_enc_correct * 100 / test_data_num

    return pln_acc, enc_acc, n_deviation


def test(data_name):
    data_name = data_name.lower()

    bits_scale = 26
    if data_name == "digits" or data_name == "mnist":
        context = ts.context( 
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=2 ** 14, 
            coeff_mod_bit_sizes=[bits_scale + 5, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale,
                                bits_scale + 5]
        )
    else:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=2 ** 13,
            coeff_mod_bit_sizes=[bits_scale + 5, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale,
                                bits_scale + 5]
        )
    context.global_scale = pow(2, bits_scale) 
    context.generate_galois_keys() 

    kernel_shape = None
    stride = None

    if data_name == "credit":
        train_loader, test_loader = load_data(data_name, batch_size=1)
        plain_model = MLP_Credit()
        plain_model.load_state_dict(torch.load(f'./pretrained/credit_plain.pt'))
        enc_model = CreditMLP_TS(plain_model)
    elif data_name == "bank":
        train_loader, test_loader = load_data(data_name, batch_size=1)
        plain_model = MLP_Bank()
        plain_model.load_state_dict(torch.load(f'./pretrained/bank_plain.pt'))
        enc_model = BankMLP_TS(plain_model)
    elif data_name == "digits":
        train_loader, test_loader = load_data(data_name, batch_size=1)
        plain_model = CryptoNet_Digits()
        plain_model.load_state_dict(torch.load(f'./pretrained/digits_plain.pt'))

        kernel_shape = plain_model.conv1.kernel_size
        stride = plain_model.conv1.stride[0]
        enc_model = DigitsCryptoNet_TS(plain_model)
    elif data_name == "mnist":
        train_loader, test_loader = load_torch_data(data_name, batch_size=1)
        plain_model = CryptoNet_MNIST()
        plain_model.load_state_dict(torch.load(f'./pretrained/mnist_plain.pt'))

        kernel_shape = plain_model.conv1.kernel_size
        stride = plain_model.conv1.stride[0]
        enc_model = MNISTCryptoNet_TS(plain_model)
    else:
        raise NotImplementedError(data_name)

    logger.info("*"*100)
    logger.info(f"TenSEAL Plaintext & Encryption Prediction")
    logger.info(f"Dataset: {data_name}, #Test Length: {len(test_loader.dataset)}")

    pln_acc, enc_acc, n_deviation = test_with_tenseal(plain_model, enc_model, test_loader, context, kernel_shape, stride)

    logger.info(f"Plaintext Prediction: {pln_acc:.2f}, Encryption Prediction: {enc_acc:.2f}, #Deviation Inputs: {n_deviation}")

    return


if __name__ == "__main__":    
    test("digits")
    test("bank")
    test("credit")
    test("mnist")
