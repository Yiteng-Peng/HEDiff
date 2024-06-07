import logging
import datetime
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plain_models import MLP_Credit, MLP_Bank, CryptoNet_Digits, CryptoNet_MNIST
from concrete.ml.torch.compile import compile_torch_model
import numpy as np
from random import sample

from tools import load_data, load_torch_data
from base_zama import *


log_filename = datetime.datetime.now().strftime("./log/zama_test.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_with_concrete(pln_model, enc_model, test_loader, use_sim=True):
    fhe_mode = "simulate" if use_sim else "execute"
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
        data = data.numpy()
        y_enc_pred = enc_model.forward(data, fhe=fhe_mode)

        y_pln_pred = np.argmax(y_pln_pred, axis=1)
        y_enc_pred = np.argmax(y_enc_pred, axis=1)
        
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


def test(data_name, use_sim=False):
    data_name = data_name.lower()
    batch_size = 1

    if data_name == "credit":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=batch_size, example=True)
        plain_model = MLP_Credit()
    elif data_name == "bank":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=batch_size, example=True)
        plain_model = MLP_Bank()
    elif data_name == "digits":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=batch_size, example=True)
        plain_model = CryptoNet_Digits()
    elif data_name == "mnist":
        train_loader, test_loader, x_train = load_torch_data(data_name, batch_size=batch_size, example=True)
        plain_model = CryptoNet_MNIST()
    else:
        raise NotImplementedError(data_name)
    plain_model.load_state_dict(torch.load(f'./pretrained/{data_name}_plain.pt'))

    print("begin compile ...")
    n_bits, p_error = 6, 0.01
    start_compile = time.time()
    enc_model = compile_torch_model(plain_model, x_train, n_bits=n_bits, rounding_threshold_bits=min(n_bits + 2, 16),
                                    p_error=p_error)
    end_compile = time.time()
    print("compile time: %.2fs" % (end_compile - start_compile))

    logger.info("*"*100)
    logger.info(f"Concrete-ML {'Simulation' if use_sim else 'RealFHE'} Plaintext & Encryption Prediction")
    logger.info(f"Dataset: {data_name}, #Test Length: {len(test_loader.dataset)}")

    pln_acc, enc_acc, n_deviation = test_with_concrete(plain_model, enc_model, test_loader, use_sim=use_sim)

    logger.info(f"Plaintext Prediction: {pln_acc:.2f}, Encryption Prediction: {enc_acc:.2f}, #Deviation Inputs: {n_deviation}")

    return


if __name__ == "__main__":
    debug = False

    test("credit", use_sim=debug)
    test("bank", use_sim=debug)
    test("digits", use_sim=debug)
    test("mnist", use_sim=debug)
