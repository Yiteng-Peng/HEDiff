import logging
import datetime
import torch
from tqdm import tqdm
from plain_models import MLP_Credit, MLP_Bank, CryptoNet_Digits_helayers, CryptoNet_MNIST_helayers
import pyhelayers
import numpy as np

from tools import load_data, load_torch_data
from base_helayers import *

log_filename = datetime.datetime.now().strftime("./log/helayer_test.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_with_helayer(pln_model, enc_model, test_loader, context, enc_shape):
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

        _, y_pln_pred = PredictPlainVector(pln_model, data)
        _, y_enc_pred = PredictEncVector(enc_model, torch.reshape(data, enc_shape), context)
        
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

    logger.info("*"*100)
    logger.info(f"Helayers Plaintext & Encryption Prediction")
    logger.info(f"Dataset: {data_name}, #Test Length: {len(test_loader.dataset)}")

    pln_acc, enc_acc, n_deviation = test_with_helayer(plain_model, enc_model, test_loader, context, enc_shape)

    logger.info(f"Plaintext Prediction: {pln_acc:.2f}, Encryption Prediction: {enc_acc:.2f}, #Deviation Inputs: {n_deviation}")

    return


if __name__ == "__main__":
    test("credit")
    test("bank")
    test("digits")
    test("mnist")
