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


log_filename = datetime.datetime.now().strftime("./log/zama_diff.log")
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


def MarginBasedDifferentialTesting(mutation_num, seed_loader, plain_model, enc_model, use_sim=False, noise_bar = 0.05, iter_bar = 0.02):
    seedList = [(data, 0, label, 0) for data, label in seed_loader]
    trueDiffList = []
    mutationList = []
    patternDict = []

    attacks = MGPGD(plain_model, eps=iter_bar, alpha=iter_bar / 4, steps=10)

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


def AlphaNoiseData(plain_model, data, noise, alpha, noise_bar):
    noise = torch.clamp(noise * alpha, min=-noise_bar, max=noise_bar)
    noise_data = torch.clamp(data + noise, min=0, max=1)
    pred_p, label_p = PredictPlainVector(plain_model, noise_data)
    return margin_metric(pred_p), label_p


def DetermineAlpha(plain_model, data, label, noise, noise_bar = 0.05, min_gap = 0.001, max_gap = 0.002):
    
    right_alpha = 10
    left_alpha = -10
    mid_alpha = 0

    margin_right, label_right = AlphaNoiseData(plain_model, data, noise, right_alpha, noise_bar)
    margin_left, label_left = AlphaNoiseData(plain_model, data, noise, left_alpha, noise_bar)
    if label_right == label and label_left == label:
        return None
    if label_right != label and label_left != label:
        return None
    
    if label_right == label:
        if min_gap < margin_right < max_gap:
            return right_alpha
        right_alpha = mid_alpha
    else:
        if min_gap < margin_left < max_gap:
            return left_alpha
        left_alpha = mid_alpha

    # Binary Search
    max_time = 1000
    for _ in range(max_time):
        mid_alpha = (left_alpha + right_alpha) / 2
        margin_mid, label_mid = AlphaNoiseData(plain_model, data, noise, mid_alpha, noise_bar)
        if margin_mid < max_gap and label_mid == label:
            return mid_alpha
        elif label_right == label_mid:
            right_alpha = mid_alpha
        else:
            left_alpha = mid_alpha

    return 0


def SimilarAlphaPatternDifferentialTesting(patternDict, seedList, plain_model, enc_model, use_sim=False, K=5, noise_bar = 0.05):
    trueDiffList = []
    seed_pattern_idx_list = [-1 for _ in range(len(seedList))]

    pattern_data_list = [pattern_data for pattern_data, noise, label_p, label_e in patternDict]
    pattern_data_list_flat = torch.stack(pattern_data_list).view(len(pattern_data_list), -1)

    all_try = 0
    no_simliar = 0

    start_time = time.time()

    pbar = tqdm(seedList)
    seed_idx = 0
    for data, old_noise, label, mu_num in pbar:
        data_flat = data.view(1, -1)
        similarities = F.pairwise_distance(data_flat, pattern_data_list_flat, p=2)
        sorted_indices = torch.argsort(similarities, descending=False)
        
        most_similar_indices = []
        K_count = 0
        for i in sorted_indices:
            if K_count == K:
                break

            most_similar_indices.append(i.item())
            K_count += 1

        if K_count == 0:
            no_simliar += 1

        for pattern_index in most_similar_indices:
            pattern_data, pattern_noise, pattern_label_p, pattern_label_e = patternDict[pattern_index]

            alpha = DetermineAlpha(plain_model, data, label, pattern_noise, noise_bar, min_gap = 0.001, max_gap = 0.002)
            if alpha is None or alpha == 0:
                continue

            mu_num += 1
            all_try += 1

            noise = torch.clamp(pattern_noise * alpha, min=-noise_bar, max=noise_bar)
            noise_data = torch.clamp(data + noise, min=0, max=1)

            pred_p, label_p = PredictPlainVector(plain_model, noise_data)
            pred_e, label_e = PredictEncVector(enc_model, noise_data, use_sim=use_sim)

            noise = noise_data - data
            if label_p != label_e and label_p == label:
                trueDiffList.append((data.clone(), noise, label.clone(), mu_num))
                seed_pattern_idx_list[seed_idx] = (pattern_index, alpha)
                break

        seed_idx += 1

        pbar.update(1)
        pbar.set_postfix({'TAEs': len(trueDiffList), "TRY": all_try})
    print({'TAEs': len(trueDiffList)})

    end_time = time.time()
    logger.info(f"Pattern DT running time[{end_time - start_time:.2f}s], Noise Bar[{noise_bar}], K Nearest[{K}]")
    logger.info(f"Total Try[{all_try}], Normal[{len(seedList)-len(trueDiffList)}], Deviation[{len(trueDiffList)}], No Similar[{no_simliar}]")

    return trueDiffList, seed_pattern_idx_list


def Start(data_name, seed_num=800, mutation_num=4000, K_near=5, use_sim=True, noise_bar = 0.05, iter_bar = 0.02):
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
    logger.info(f"Concrete-ML {'Simulation' if use_sim else 'RealFHE'} Differential Testing Start")
    logger.info(f"Dataset: {data_name}, #Seed: {seed_num}, #Mutation: {mutation_num}, #K nearest: {K_near}")

    # step 1: seed filter
    logger.info(f"Step 1: Seed Filtering")
    seed_loader = mertric_sort(seed_num, plain_model, train_loader)

    # step 1.1: without mutation, just check
    _, oriTrueDiffList, sameList = OriDifferentialTesting(seed_loader, plain_model, enc_model, use_sim=use_sim)
    
    # step 2: Margin-based mutation
    logger.info(f"Step 2: Mutation")
    muTrueDiffList, muSameList, patternDict, mutationList = MarginBasedDifferentialTesting(mutation_num, sameList, plain_model, enc_model, use_sim=use_sim, noise_bar = noise_bar, iter_bar = iter_bar)

    # step 3: noise pattern
    
    if len(patternDict) == 0:
        logger.warning(f"No deviation in Step 2, skip Step 3")
        patTrueDiffList, seed_pattern_idx_list = [], []
    else:
        logger.info(f"Step 3: Pattern")
        patTrueDiffList, seed_pattern_idx_list = SimilarAlphaPatternDifferentialTesting(patternDict, muSameList, plain_model, enc_model, use_sim=use_sim, K=K_near, noise_bar = noise_bar)

    result_tuple = (sameList, muSameList, oriTrueDiffList, muTrueDiffList, patTrueDiffList, patternDict, seed_pattern_idx_list, mutationList)

    # exp step: important file save
    pkl_filename = datetime.datetime.now().strftime(f"./corpus/zama_{data_name}(%m%d-%H%M%S).pkl")
    pkl_filename = f"./corpus/zama_{data_name}.pkl"
    with open(pkl_filename, 'wb') as fp:
        pickle.dump(result_tuple, fp)

    logger.info(f"File save in {pkl_filename}")

    return result_tuple


if __name__ == "__main__":
    debug = False

    Start("credit", use_sim=debug, seed_num=500, mutation_num=2500, K_near=1, noise_bar = 0.01, iter_bar=0.003)
    Start("bank", use_sim=debug, seed_num=500, mutation_num=2500, K_near=1, noise_bar = 0.01, iter_bar=0.003)
    Start("digits", use_sim=debug, seed_num=250, mutation_num=1250, K_near=1, noise_bar = 0.05, iter_bar=0.03)
    Start("mnist", use_sim=debug, seed_num=500, mutation_num=2500, K_near=1, noise_bar = 0.03, iter_bar=0.01)

    print("diff_zama")
