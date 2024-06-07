import logging
import pickle
import datetime
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plain_models import MLP_Credit, MLP_Bank, CryptoNet_Digits, CryptoNet_MNIST
from concrete.ml.torch.compile import compile_torch_model
from torchattacks.attack import Attack
from random import sample

from tools import load_data, load_torch_data
from base_zama import *
from base_margin import *


log_filename = datetime.datetime.now().strftime("./log/zama_mu.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def OriDifferentialTesting(seed_loader, plain_model, enc_model, use_sim=False):
    seedList = [(data, label) for data, label in seed_loader]
    trueDiffList = []
    falseDiffList = []
    sameList = []

    start_time = time.time()

    pbar = tqdm(seedList)
    for data, label in pbar:
        _, label_p = PredictPlainVector(plain_model, data)
        _, label_e = PredictEncVector(enc_model, data, use_sim=use_sim)

        if label_p != label_e:
            if label_p == label:
                trueDiffList.append((data, label))
            else:
                falseDiffList.append((data, label))
        else:
            sameList.append((data, label))

        pbar.set_postfix({'FNum': len(falseDiffList), 'FRatio': len(falseDiffList) * 100.0 / len(seedList),
                          'TNum': len(trueDiffList), 'TRatio': len(trueDiffList) * 100.0 / len(seedList)})

    print(f'FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)', end=" | ")
    print(f'TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)')

    end_time = time.time()
    logger.info("Origin DT running time: %.2fs" % (end_time - start_time))
    logger.info(f"FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)")
    logger.info(f"TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)")

    return falseDiffList, trueDiffList, sameList


def MarginBasedDifferentialTesting(mutation_method, mutation_num, seed_loader, plain_model, enc_model, use_sim=False, noise_bar = 0.05, iter_bar = 0.02):
    seedList = [(data, 0, label, 0) for data, label in seed_loader]
    trueDiffList = []
    mutationList = []
    patternDict = []

    attacks = MGPGD_mu(mutation_method, plain_model, eps=iter_bar, alpha=iter_bar / 4, steps=10)

    start_time = time.time()

    total_mutation = 0
    pbar = tqdm(total=mutation_num)
    while total_mutation < mutation_num and len(seedList) > 0:
        data, old_noise, label, mu_num = seedList.pop(0)
        mu_num += 1

        noise = attacks.forward(data + old_noise)   
        noise = old_noise + noise
        noise = torch.clamp(noise, min=-noise_bar, max=noise_bar)
        noise_data = torch.clamp(data + noise, min=0, max=1)

        _, label_p = PredictPlainVector(plain_model, noise_data)
        _, label_e = PredictEncVector(enc_model, noise_data, use_sim=use_sim)

        noise = noise_data - data
        if label_p != label_e and label_p == label:
            trueDiffList.append((data.clone(), noise.clone(), label.clone(), mu_num))
            mutationList.append(total_mutation)
            patternDict.append((data, noise, label_p, label_e))
        else:
            seedList.append((data, noise, label, mu_num))

        total_mutation += 1
        pbar.update(1)
        pbar.set_postfix({'TAEs': len(trueDiffList), 'Mutation': total_mutation})
    print({'TAEs': len(trueDiffList), 'Mutation': total_mutation})

    end_time = time.time()
    logger.info(f"Mutation DT running time[{end_time - start_time:.2f}s], Noise Bar[{noise_bar}], Iter Bar[{iter_bar}]")
    logger.info(f"Total Mutation[{total_mutation}], Normal[{len(seedList)}], Deviation[{len(trueDiffList)}]")

    return trueDiffList, seedList, patternDict, mutationList


def Start(data_name, seed_filter, mutation_method, seed_num=800, mutation_num=4000, noise_bar = 0.05, iter_bar = 0.02, use_sim=False):
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
    print(f"compile time: {end_compile - start_compile:.2f}s")

    logger.info("="*100)
    logger.info(f"Concrete-ML {'Simulation' if use_sim else 'RealFHE'} Mutation Start")
    logger.info(f"Dataset: {data_name}, #Mutation: {mutation_num}, Seed Length: {seed_num}")

    # step 1: seed filter
    logger.info(f"Step 1: {seed_filter} Seed Filtering")
    if seed_filter == "margin":
        seed_loader = mertric_sort(seed_num, plain_model, train_loader)
    elif seed_filter == "random":
        seed_loader = [(data, label) for data, label in train_loader]
        seed_loader = sample(seed_loader, seed_num)
    _, oriTrueDiffList, sameList = OriDifferentialTesting(seed_loader, plain_model, enc_model, use_sim=use_sim)

    # severial mutation
    logger.info(f"Step 2: {mutation_method} Mutation")
    MarginBasedDifferentialTesting(mutation_method, mutation_num, sameList, plain_model, enc_model, use_sim=use_sim, noise_bar = noise_bar, iter_bar = iter_bar)

    return


if __name__ == "__main__":
    Start("credit", "random", "random", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    Start("credit", "margin", "random", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    Start("credit", "random", "margin", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    Start("credit", "margin", "margin", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)

    
    Start("bank", "random", "random", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    Start("bank", "margin", "random", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    Start("bank", "random", "margin", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    Start("bank", "margin", "margin", seed_num=100, mutation_num=300, noise_bar = 0.01, iter_bar=0.003)
    
    Start("digits", "random", "random", seed_num=100, mutation_num=300, noise_bar = 0.05, iter_bar = 0.03)
    Start("digits", "margin", "random", seed_num=100, mutation_num=300, noise_bar = 0.05, iter_bar = 0.03)
    Start("digits", "random", "margin", seed_num=100, mutation_num=300, noise_bar = 0.05, iter_bar = 0.03)
    Start("digits", "margin", "margin", seed_num=100, mutation_num=300, noise_bar = 0.05, iter_bar = 0.03)

    Start("mnist", "random", "random", seed_num=100, mutation_num=300, noise_bar = 0.03, iter_bar = 0.01)
    Start("mnist", "margin", "random", seed_num=100, mutation_num=300, noise_bar = 0.03, iter_bar = 0.01)
    Start("mnist", "random", "margin", seed_num=100, mutation_num=300, noise_bar = 0.03, iter_bar = 0.01)
    Start("mnist", "margin", "margin", seed_num=100, mutation_num=300, noise_bar = 0.03, iter_bar = 0.01)

    print("mu_zama")
