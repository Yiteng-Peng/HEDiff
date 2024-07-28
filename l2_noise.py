import logging
import pickle
import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample


log_filename = datetime.datetime.now().strftime("./log/l2_noise.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def l2Value(filename):
    pkl_filename = f"./corpus/{filename}.pkl"
    with open(pkl_filename, 'rb') as fp:
        result_tuple = pickle.load(fp)

    sameList, muSameList, oriTrueDiffList, muTrueDiffList, patTrueDiffList, patternDict, seed_pattern_idx_list, mutationList = result_tuple

    noise_l2_list = [torch.norm(noise, p=2).item() / noise.numel() for pattern_data, noise, label_p, label_e in patternDict]
    noise_max_list = [torch.max(torch.abs(noise)).item() for pattern_data, noise, label_p, label_e in patternDict]

    print((f"l2: {filename}, [{(sum(noise_l2_list) / len(noise_l2_list)):.4f}], length: {len(noise_l2_list)}"))
    logger.info(f"{filename}, length: {len(noise_l2_list)}. AVG L2: [{(sum(noise_l2_list) / len(noise_l2_list)):.4f}]. AVG MAX: [{(sum(noise_max_list) / len(noise_max_list)):.4f}]")


if __name__ == "__main__":
    # Table 5
    l2Value("ts_mnist")
    l2Value("ts_digits")
    l2Value("ts_credit")
    l2Value("ts_bank")

    l2Value("zama_mnist")
    l2Value("zama_digits")
    l2Value("zama_credit")
    l2Value("zama_bank")

    l2Value("helayer_mnist")
    l2Value("helayer_digits")
    l2Value("helayer_bank")
