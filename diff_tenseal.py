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

from base_ts import *
from base_margin import *


log_filename = datetime.datetime.now().strftime("./log/ts_diff.log")
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
                trueDiffList.append((data, label))
            else:
                falseDiffList.append((data, label))
        else:
            sameList.append((data, label))

        pbar.set_postfix({'FNum': len(falseDiffList), 'FRatio': len(falseDiffList) * 100.0 / len(seedList),
                          'TNum': len(trueDiffList), 'TRatio': len(trueDiffList) * 100.0 / len(seedList)})

    print(f'FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)')
    print(f'TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)')

    end_time = time.time()
    logger.info("Origin DT running time: %.2fs" % (end_time - start_time))
    logger.info(f"FNum: {len(falseDiffList)}/{len(seedList)}({len(falseDiffList) * 100.0 / len(seedList):.2f}%)")
    logger.info(f"TNum: {len(trueDiffList)}/{len(seedList)}({len(trueDiffList) * 100.0 / len(seedList):.2f}%)")

    return falseDiffList, trueDiffList, sameList


class MGPGD(Attack):
    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__('MGPGD', model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels=None):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        alpha = self.alpha

        if self.random_start:
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            cost =  -1 * margin_metric(outputs)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            alpha = alpha / 2
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        noise_images = adv_images - images

        return noise_images


def MarginBasedDifferentialTesting(mutation_num, seed_loader, plain_model, enc_model, context, 
                                   kernel_shape=None, stride=None, noise_bar = 0.05, iter_bar = 0.02):
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
        if kernel_shape is not None:
            _, label_e = PredictConvEncVector(enc_model, noise_data, context, kernel_shape, stride)
        else:
            _, label_e = PredictEncVector(enc_model, noise_data, context)

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


def SimilarAlphaPatternDifferentialTesting(patternDict, seedList, plain_model, enc_model, context, 
                                           kernel_shape=None, stride=None, K=5, noise_bar = 0.05, same_label=True):
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

            _, label_p = PredictPlainVector(plain_model, noise_data)
            if kernel_shape is not None:
                _, label_e = PredictConvEncVector(enc_model, noise_data, context, kernel_shape, stride)
            else:
                _, label_e = PredictEncVector(enc_model, noise_data, context)

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


def Start(data_name, seed_num=800, mutation_num=4000, K_near=5, noise_bar = 0.05, iter_bar = 0.02):
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
    
    logger.info("="*100)
    logger.info(f"TenSEAL Differential Testing Start")
    logger.info(f"Dataset: {data_name}, #Seed: {seed_num}, #Mutation: {mutation_num}, #K nearest: {K_near}")

    # step 1: seed filter
    logger.info(f"Step 1: Seed Filtering")
    seed_loader = mertric_sort(seed_num, plain_model, train_loader)

    # step 1.1: without mutation, just check
    _, oriTrueDiffList, sameList = OriDifferentialTesting(seed_loader, plain_model, enc_model, context, kernel_shape, stride)
    
    # step 2: Margin-based mutation
    logger.info(f"Step 2: Mutation")
    muTrueDiffList, muSameList, patternDict, mutationList = MarginBasedDifferentialTesting(mutation_num, sameList, plain_model, enc_model, context, kernel_shape, stride, noise_bar = noise_bar, iter_bar = iter_bar)

    # step 3: noise pattern
    if len(patternDict) == 0:
        logger.warning(f"No deviation in Step 2, skip Step 3")
        patTrueDiffList, seed_pattern_idx_list = [], []
    else:
        logger.info(f"Step 3: Pattern")
        patTrueDiffList, seed_pattern_idx_list = SimilarAlphaPatternDifferentialTesting(patternDict, muSameList, plain_model, enc_model, context, kernel_shape, stride, K=K_near, noise_bar = noise_bar)

    result_tuple = (sameList, muSameList, oriTrueDiffList, muTrueDiffList, patTrueDiffList, patternDict, seed_pattern_idx_list, mutationList)

    # exp step: important file save
    pkl_filename = f"./corpus/ts_{data_name}.pkl"
    with open(pkl_filename, 'wb') as fp:
        pickle.dump(result_tuple, fp)

    logger.info(f"File save in {pkl_filename}")

    return result_tuple

if __name__ == "__main__":
    Start("credit", seed_num=1000, mutation_num=5000, K_near=1, noise_bar = 0.05, iter_bar=0.03)
    Start("bank", seed_num=1000, mutation_num=5000, K_near=1, noise_bar = 0.03, iter_bar=0.01)
    Start("digits", seed_num=500, mutation_num=2500, K_near=1, noise_bar = 0.05, iter_bar=0.03)
    Start("mnist", seed_num=1000, mutation_num=5000, K_near=1, noise_bar = 0.05, iter_bar=0.03)

    print("diff_tenseal")