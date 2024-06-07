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
from tqdm import tqdm
import tenseal as ts
from torchattacks.attack import Attack
from random import sample


from base_ts import *
from base_margin import margin_metric


log_filename = datetime.datetime.now().strftime("./log/ts_pattern.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


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


def SimilarAlphaPatternDifferentialTesting(label_tag, close_tag, patternDict, seedList, plain_model, enc_model, context, kernel_shape=None, stride=None, noise_bar = 0.05):
    maxK = 5
    
    trueDiffList_k1 = []
    try_k1 = 0
    trueDiffList_k3 = []
    try_k3 = 0
    trueDiffList_k5 = []
    try_k5 = 0

    pattern_data_list = [pattern_data for pattern_data, noise, label_p, label_e in patternDict]
    pattern_data_list_flat = torch.stack(pattern_data_list).view(len(pattern_data_list), -1)

    start_time = time.time()

    pbar = tqdm(seedList)
    for data, old_noise, label, mu_num in pbar:
        data_flat = data.view(1, -1)
        if close_tag == "topk":
            similarities = F.pairwise_distance(data_flat, pattern_data_list_flat, p=2)
            sorted_indices = torch.argsort(similarities, descending=False)
        elif close_tag == "random":
            sorted_indices = sample(range(len(pattern_data_list_flat)), len(pattern_data_list_flat))
            sorted_indices = torch.Tensor(sorted_indices).to(torch.int)
        else:
            raise NotImplementedError
        
        most_similar_indices = []
        K_count = 0
        for i in sorted_indices:
            if K_count == maxK:
                break
                
            if label_tag == True:
                if patternDict[i][2] == label:
                    most_similar_indices.append(i.item())
                    K_count += 1
            else:
                most_similar_indices.append(i.item())
                K_count += 1
        
        top1_tag, top3_tag = True, True
        idx_cnt = 0
        for pattern_index in most_similar_indices:
            
            pattern_data, pattern_noise, pattern_label_p, pattern_label_e = patternDict[pattern_index]
            alpha = DetermineAlpha(plain_model, data, label, pattern_noise, noise_bar, min_gap = 0.001,max_gap = 0.002)

            if alpha is None or alpha == 0:
                continue
            
            mu_num += 1
            
            if top1_tag:
                try_k1 += 1
            if top3_tag:
                try_k3 += 1
            try_k5 += 1
            
            noise = torch.clamp(pattern_noise * alpha, min=-noise_bar, max=noise_bar)
            noise_data = torch.clamp(data + noise, min=0, max=1)
            
            _, label_p = PredictPlainVector(plain_model, noise_data)
            if kernel_shape is not None:
                _, label_e = PredictConvEncVector(enc_model, noise_data, context, kernel_shape, stride)
            else:
                _, label_e = PredictEncVector(enc_model, noise_data, context)
            
            noise = noise_data - data
            if label_p != label_e and label_p == label:
                if top1_tag:
                    trueDiffList_k1.append((data.clone(), noise, label.clone(), mu_num))
                if top3_tag:
                    trueDiffList_k3.append((data.clone(), noise, label.clone(), mu_num))
                trueDiffList_k5.append((data.clone(), noise, label.clone(), mu_num))

                break

            idx_cnt += 1
            if idx_cnt == 1:
                top1_tag = False 
            if idx_cnt == 3:
                top3_tag = False
    
        pbar.update(1)
        pbar.set_postfix({'1TAEs': len(trueDiffList_k1), "1TRY": try_k1, 
                          '3TAEs': len(trueDiffList_k3), "3TRY": try_k3,
                          '5TAEs': len(trueDiffList_k5), "5TRY": try_k5})

    end_time = time.time()
    logger.info(f"Pattern DT running time[{end_time - start_time:.2f}s], Noise Bar[{noise_bar}], K Nearest[1, 3, 5]")
    logger.info(f"Try 1[{try_k1}], Normal[{len(seedList)-len(trueDiffList_k1)}], Deviation[{len(trueDiffList_k1)}]]")
    logger.info(f"Try 3[{try_k3}], Normal[{len(seedList)-len(trueDiffList_k3)}], Deviation[{len(trueDiffList_k3)}]]")
    logger.info(f"Try 5[{try_k5}], Normal[{len(seedList)-len(trueDiffList_k5)}], Deviation[{len(trueDiffList_k5)}]]")

    return


def PatternEffect(data_name, filename, label_tag, close_tag, noise_bar):
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
        plain_model = MLP_Credit()
        plain_model.load_state_dict(torch.load(f'./pretrained/credit_plain.pt'))
        enc_model = CreditMLP_TS(plain_model)

    elif data_name == "bank":
        plain_model = MLP_Bank()
        plain_model.load_state_dict(torch.load(f'./pretrained/bank_plain.pt'))
        enc_model = BankMLP_TS(plain_model)

    elif data_name == "digits":
        plain_model = CryptoNet_Digits()
        plain_model.load_state_dict(torch.load(f'./pretrained/digits_plain.pt'))

        kernel_shape = plain_model.conv1.kernel_size
        stride = plain_model.conv1.stride[0]
        enc_model = DigitsCryptoNet_TS(plain_model)

    elif data_name == "mnist":
        plain_model = CryptoNet_MNIST()
        plain_model.load_state_dict(torch.load(f'./pretrained/mnist_plain.pt'))

        kernel_shape = plain_model.conv1.kernel_size
        stride = plain_model.conv1.stride[0]
        enc_model = MNISTCryptoNet_TS(plain_model)
    
    else:
        raise NotImplementedError(data_name)

    pkl_filename = f"./corpus/{filename}.pkl"
    with open(pkl_filename, 'rb') as fp:
        result_tuple = pickle.load(fp)

    sameList, muSameList, oriTrueDiffList, muTrueDiffList, patTrueDiffList, patternDict, seed_pattern_idx_list, mutationList = result_tuple

    logger.info("%"*100)
    logger.info(f"TenSEAL Pattern Start, Dataset:{filename}, Seed Length: {len(muSameList)}, Pattern Length: {len(muTrueDiffList)}")

    SimilarAlphaPatternDifferentialTesting(label_tag, close_tag, patternDict, muSameList, plain_model, enc_model, 
                                           context, kernel_shape, stride, noise_bar = noise_bar)

    return


if __name__ == "__main__":
    PatternEffect("mnist", "ts_mnist_large", True, "topk", noise_bar = 0.05)
    PatternEffect("mnist", "ts_mnist_large", False, "topk", noise_bar = 0.05)
    PatternEffect("mnist", "ts_mnist_large", True, "random", noise_bar = 0.05)
    PatternEffect("mnist", "ts_mnist_large", False, "random", noise_bar = 0.05)

    PatternEffect("digits", "ts_digits_large", True, "topk", noise_bar = 0.05)
    PatternEffect("digits", "ts_digits_large", False, "topk", noise_bar = 0.05)
    PatternEffect("digits", "ts_digits_large", True, "random", noise_bar = 0.05)
    PatternEffect("digits", "ts_digits_large", False, "random", noise_bar = 0.05)

    PatternEffect("bank", "ts_bank", True, "topk", noise_bar = 0.03)
    PatternEffect("bank", "ts_bank", False, "topk", noise_bar = 0.03)
    PatternEffect("bank", "ts_bank", True, "random", noise_bar = 0.03)
    PatternEffect("bank", "ts_bank", False, "random", noise_bar = 0.03)

    print("pattern_tenseal")
