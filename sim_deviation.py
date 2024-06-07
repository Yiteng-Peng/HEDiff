import logging
import pickle
import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample


log_filename = datetime.datetime.now().strftime("./log/deviation_sim.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def SimValue(filename, label_tag, close_tag, K):
    pkl_filename = f"./corpus/{filename}.pkl"
    with open(pkl_filename, 'rb') as fp:
        result_tuple = pickle.load(fp)

    sameList, muSameList, oriTrueDiffList, muTrueDiffList, patTrueDiffList, patternDict, seed_pattern_idx_list, mutationList = result_tuple

    pattern_data_list = [pattern_data for pattern_data, noise, label_p, label_e in patternDict]
    pattern_data_list_flat = torch.stack(pattern_data_list).view(len(pattern_data_list), -1)

    all_similarity = 0.0
    empty_cnt = 0

    pbar = tqdm(patternDict)
    noise_idx = 0
    for data, old_noise, label, mu_num in pbar:
        data_flat = data.view(1, -1)

        # calculate distance
        if close_tag == "topk":
            similarities = F.pairwise_distance(data_flat, pattern_data_list_flat, p=2)
            sorted_indices = torch.argsort(similarities, descending=False)
        elif close_tag == "random":
            sorted_indices = sample(range(len(pattern_data_list_flat)), len(pattern_data_list_flat))
            sorted_indices = torch.Tensor(sorted_indices).to(torch.int)
        else:
            raise NotImplementedError
        
        # get similar indices
        most_similar_indices = []
        K_count = 0
        for i in sorted_indices:
            if K_count == K:
                break
            if i == noise_idx:
                continue

            if label_tag == True:
                if patternDict[i][2] == label:
                    most_similar_indices.append(i.item())
                    K_count += 1
            else:
                most_similar_indices.append(i.item())
                K_count += 1

        # calculate average cosine similarity
        item_average_similarity = 0
        for sim_idx in most_similar_indices:
            item_average_similarity += F.cosine_similarity(data_flat, pattern_data_list_flat[sim_idx], dim=1).item()
        item_average_similarity = item_average_similarity / len(most_similar_indices)

        if len(most_similar_indices) != 0:
            all_similarity += item_average_similarity
        else:
            empty_cnt += 1

        noise_idx += 1

    all_similarity = all_similarity / (len(patternDict) - empty_cnt)

    print(f"Similarity Analysis: {filename}, Label: {label_tag}, Close: {close_tag}, K: {K}, Sim: {all_similarity:.4f}, No Sim: {empty_cnt}")
    logger.info(f"Similarity Analysis: {filename}, Label: {label_tag}, Close: {close_tag}, K: {K}, Sim: {all_similarity:.4f}, No Sim: {empty_cnt}")


if __name__ == "__main__":
    # Table 6
    SimValue("ts_mnist_large", True, "topk", 1)
    SimValue("ts_mnist_large", False, "topk", 1)
    SimValue("ts_mnist_large", True, "random", 1)
    SimValue("ts_mnist_large", False, "random", 1)

    SimValue("ts_mnist_large", True, "topk", 3)
    SimValue("ts_mnist_large", False, "topk", 3)
    SimValue("ts_mnist_large", True, "random", 3)
    SimValue("ts_mnist_large", False, "random", 3)

    SimValue("ts_mnist_large", True, "topk", 5)
    SimValue("ts_mnist_large", False, "topk", 5)
    SimValue("ts_mnist_large", True, "random", 5)
    SimValue("ts_mnist_large", False, "random", 5)

    SimValue("ts_digits_large", True, "topk", 1)
    SimValue("ts_digits_large", False, "topk", 1)
    SimValue("ts_digits_large", True, "random", 1)
    SimValue("ts_digits_large", False, "random", 1)

    SimValue("ts_digits_large", True, "topk", 3)
    SimValue("ts_digits_large", False, "topk", 3)
    SimValue("ts_digits_large", True, "random", 3)
    SimValue("ts_digits_large", False, "random", 3)

    SimValue("ts_digits_large", True, "topk", 5)
    SimValue("ts_digits_large", False, "topk", 5)
    SimValue("ts_digits_large", True, "random", 5)
    SimValue("ts_digits_large", False, "random", 5)

    SimValue("ts_bank", True, "topk", 1)
    SimValue("ts_bank", False, "topk", 1)
    SimValue("ts_bank", True, "random", 1)
    SimValue("ts_bank", False, "random", 1)

    SimValue("ts_bank", True, "topk", 3)
    SimValue("ts_bank", False, "topk", 3)
    SimValue("ts_bank", True, "random", 3)
    SimValue("ts_bank", False, "random", 3)

    SimValue("ts_bank", True, "topk", 5)
    SimValue("ts_bank", False, "topk", 5)
    SimValue("ts_bank", True, "random", 5)
    SimValue("ts_bank", False, "random", 5)
