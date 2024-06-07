import logging
import pickle
import datetime
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plain_models import MLP_Credit, MLP_Bank, CryptoNet_Digits_helayers, CryptoNet_MNIST_helayers
import pyhelayers
from random import sample

from tools import load_data, load_torch_data
from base_helayers import *

log_filename = datetime.datetime.now().strftime("./log/helayer_ori.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def OriDifferentialTesting(seed_loader, plain_model, enc_model, context, enc_shape):
    seedList = [(data, label) for data, label in seed_loader]
    trueDiffList = []
    falseDiffList = []
    sameList = []

    start_time = time.time()

    pbar = tqdm(seedList)
    for data, label in pbar:
        pred_p, label_p = PredictPlainVector(plain_model, data)
        pred_e, label_e = PredictEncVector(enc_model, torch.reshape(data, enc_shape), context)

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
        enc_shape = (-1, 23)
        plain_model.load_state_dict(torch.load(f'./pretrained/{data_name}_plain.pt'))
    elif data_name == "bank":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = MLP_Bank()
        enc_shape = (-1, 20)
        plain_model.load_state_dict(torch.load(f'./pretrained/{data_name}_plain.pt'))
    elif data_name == "digits":
        train_loader, test_loader, x_train = load_data(data_name, batch_size=1, example=True)
        plain_model = CryptoNet_Digits_helayers()
        enc_shape = (-1, 8, 8, 1)
        plain_model.load_state_dict(torch.load(f'./pretrained/{data_name}_plain_tf.pt'))
    elif data_name == "mnist":
        train_loader, test_loader, x_train = load_torch_data(data_name, batch_size=1, example=True)
        plain_model = CryptoNet_MNIST_helayers()
        enc_shape = (-1, 28, 28, 1)
        plain_model.load_state_dict(torch.load(f'./pretrained/{data_name}_plain_tf.pt'))
    else:
        raise NotImplementedError(data_name)


    hyper_params = pyhelayers.PlainModelHyperParams()
    enc_model_p = pyhelayers.NeuralNetPlain()

    if data_name == "credit" or data_name == "bank":
        enc_model_p.init_from_files(hyper_params, [f'./pretrained/{data_name}_plain.onnx'])
    elif data_name == "digits" or data_name == "mnist":
        enc_model_p.init_from_files(hyper_params, [f"./pretrained/{data_name}_plain_tf.json", f"./pretrained/{data_name}_plain_tf.h5"])

    he_run_req = pyhelayers.HeRunRequirements()
    he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
    he_run_req.optimize_for_batch_size(1)

    profile = pyhelayers.HeModel.compile(enc_model_p, he_run_req)
    context = pyhelayers.HeModel.create_context(profile)

    enc_model = pyhelayers.NeuralNet(context)
    enc_model.encode_encrypt(enc_model_p, profile)

    seed_loader = [(data, label) for data, label in train_loader]
    if seed_num is not None:
        seed_loader = sample(seed_loader, seed_num)

    logger.info("*"*100)
    logger.info(f"Helyaers Origin Differential Testing Start")
    logger.info(f"Dataset: {data_name}, #Seed: {len(seed_loader)}")

    falseDiffList, trueDiffList, sameList = OriDifferentialTesting(seed_loader, plain_model, enc_model, context, enc_shape)

    result_tuple = falseDiffList, trueDiffList, sameList 
    pkl_filename = f"./corpus/ori_helayer_{data_name}.pkl"
    with open(pkl_filename, 'wb') as fp:
        pickle.dump(result_tuple, fp)
    
    return


if __name__ == "__main__":
    OriStart("credit")
    OriStart("bank")
    OriStart("digits")
    OriStart("mnist", seed_num=5000)
