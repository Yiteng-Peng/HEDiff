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


log_filename = datetime.datetime.now().strftime("./log/ts_ori.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def OriDifferentialTesting(seed_loader, plain_model, enc_model, context, kernel_shape=None, stride=None):
    seedList = [(data, label) for data, label in seed_loader]
    trueDiffList = []
    falseDiffList = []
    sameList = []

    start_time = time.time()

    pbar = tqdm(seedList)
    for data, label in pbar:
        pred_p, label_p = PredictPlainVector(plain_model, data)
        if kernel_shape is not None:
            pred_e, label_e = PredictConvEncVector(enc_model, data, context, kernel_shape, stride)
        else:
            pred_e, label_e = PredictEncVector(enc_model, data, context)

        if label_p != label_e:
            if label_p == label:
                trueDiffList.append((data, label, pred_p, pred_e))
            else:
                falseDiffList.append((data, label, pred_p, pred_e))
        else:
            sameList.append((data, label, pred_p, pred_e))

        pbar.set_postfix({'FNum': len(falseDiffList), 'FRatio': len(falseDiffList) * 100.0 / len(seedList),
                          'TNum': len(trueDiffList), 'TRatio': len(trueDiffList) * 100.0 / len(seedList)})

    print(f'FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)')
    print(f'TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)')

    end_time = time.time()
    logger.info("Origin DT running time: %.2fs" % (end_time - start_time))
    logger.info(f"FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)")
    logger.info(f"TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)")

    return falseDiffList, trueDiffList, sameList


def OriStart(data_name, seed_num=None):
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
    context.global_scale = pow(2, bits_scale)  # set the scale
    context.generate_galois_keys()  # galois keys are required to do ciphertext rotations
        
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

    seed_loader = [(data, label) for data, label in train_loader]
    if seed_num is not None:
        seed_loader = sample(seed_loader, seed_num)

    logger.info("*"*100)
    logger.info(f"TenSEAL Origin Differential Testing Start")
    logger.info(f"Dataset: {data_name}, #Seed: {len(seed_loader)}")

    falseDiffList, trueDiffList, sameList = OriDifferentialTesting(seed_loader, plain_model, enc_model, context, kernel_shape, stride)
    
    result_tuple = falseDiffList, trueDiffList, sameList 
    pkl_filename = f"./corpus/ori_ts_{data_name}.pkl"
    with open(pkl_filename, 'wb') as fp:
        pickle.dump(result_tuple, fp)
    
    return


if __name__ == "__main__":
    OriStart("credit")
    OriStart("bank")
    OriStart("digits")
    OriStart("mnist", seed_num=5000)
