import logging
import pickle
import datetime
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plain_models import MLP_Credit, MLP_Bank, CryptoNet_Digits, CryptoNet_MNIST
from concrete.ml.torch.compile import compile_torch_model
from random import sample


from tools import load_data, load_torch_data
from base_zama import *


log_filename = datetime.datetime.now().strftime("./log/zama_ori.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def OriDifferentialTesting(seed_loader, plain_model, enc_model, use_sim=True):
    seedList = [(data, label) for data, label in seed_loader]
    trueDiffList = []
    falseDiffList = []
    sameList = []

    start_time = time.time()

    pbar = tqdm(seedList)
    for data, label in pbar:
        pred_p, label_p = PredictPlainVector(plain_model, data)
        pred_e, label_e = PredictEncVector(enc_model, data, use_sim=use_sim)

        if label_p != label_e:
            if label_p == label:
                trueDiffList.append((data, label, pred_p, pred_e))
            else:
                falseDiffList.append((data, label, pred_p, pred_e))
        else:
            sameList.append((data, label, pred_p, pred_e))

        pbar.set_postfix({'FNum': len(falseDiffList), 'FRatio': len(falseDiffList) * 100.0 / len(seedList),
                          'TNum': len(trueDiffList), 'TRatio': len(trueDiffList) * 100.0 / len(seedList)})

    print(f'FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)', end=" | ")
    print(f'TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)')

    end_time = time.time()
    logger.info("Origin DT running time: %.2fs" % (end_time - start_time))
    logger.info(f"FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)")
    logger.info(f"TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)")

    return falseDiffList, trueDiffList, sameList


def OriStart(data_name, use_sim=False, seed_num=None):
    data_name = data_name.lower()

    if data_name == "credit":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = MLP_Credit()
    elif data_name == "bank":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = MLP_Bank()
    elif data_name == "digits":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = CryptoNet_Digits()
    elif data_name == "mnist":
        train_loader, test_loader, x_train = load_torch_data(data_name, batch_size=1, example=True)
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

    seed_loader = [(data, label) for data, label in train_loader]
    if seed_num is not None:
        seed_loader = sample(seed_loader, seed_num)

    logger.info("*"*100)
    logger.info(f"Concrete-ML {'Simulation' if use_sim else 'RealFHE'} Origin Differential Testing Start")
    logger.info(f"Dataset: {data_name}, #Seed: {len(seed_loader)}")

    falseDiffList, trueDiffList, sameList = OriDifferentialTesting(seed_loader, plain_model, enc_model, use_sim=use_sim)

    result_tuple = falseDiffList, trueDiffList, sameList 
    pkl_filename = f"./corpus/ori_zama_{'sim' if use_sim else 'fhe'}_{data_name}.pkl"
    with open(pkl_filename, 'wb') as fp:
        pickle.dump(result_tuple, fp)
    
    return


if __name__ == "__main__":
    debug = False

    OriStart("credit", use_sim=debug)
    OriStart("bank", use_sim=debug)
    OriStart("digits", use_sim=debug)
    OriStart("mnist", use_sim=debug, seed_num=5000)
