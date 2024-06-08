import logging
import pickle
import datetime
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample


log_filename = datetime.datetime.now().strftime("./log/deviation.log")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S',
                    filename=log_filename, filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def secondClassRatio(filename):
    pkl_filename = f"./corpus/{filename}.pkl"
    with open(pkl_filename, 'rb') as fp:
        result_tuple = pickle.load(fp)

    falseDiffList, trueDiffList, sameList = result_tuple

    cnt_second = 0
    for data, label, pred_p, pred_e in trueDiffList:
        values, indices = torch.topk(pred_p, k=2)
        second_plain_label = indices[0][1]
        fhe_label = pred_e.argmax(1)

        if second_plain_label == fhe_label:
            cnt_second += 1

    print(f'{cnt_second}/{len(trueDiffList)}({cnt_second * 100.0 / len(trueDiffList):.2f}%)')

    logger.info(f"Second Class Analysis: {filename}")
    logger.info(f'{cnt_second}/{len(trueDiffList)}({cnt_second * 100.0 / len(trueDiffList):.2f}%)')


def margin(pred):
    topk_values, _ = torch.topk(pred, k=2)
    metric_value = topk_values[0][0] - topk_values[0][1]  # largest - second largest
    return metric_value.item()


def get_metric_list(diffList, sameList, max_num=None):
    diff_metric_list = []
    same_metric_list = []
    
    for data, label, pred_p, pred_e in tqdm(diffList):
        diff_metric = margin(pred_p)
        diff_metric_list.append(diff_metric)

    if max_num is not None:
        sample_sameList = sample(sameList, max_num)
    else:
        sample_sameList = sameList

    for data, label, pred_p, pred_e in tqdm(sample_sameList):
        same_metric = margin(pred_p)
        same_metric_list.append(same_metric)

    return diff_metric_list, same_metric_list


def draw_diff(diff_metric_list, same_metric_list):
    plt.rcParams["font.family"] = "Times New Roman"

    plt.hist(diff_metric_list, bins=5, color='blue', alpha=0.3, label="Deviation")
    plt.hist(same_metric_list, bins=50, color='red', alpha=0.3, label="Normal")

    plt.title("Margin Value Comparison", fontsize=20)
    plt.yscale('log')
    plt.xlabel('Margin Value', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.yticks(size = 18)
    plt.xticks(size = 18)
    plt.legend(fontsize=18)
    plt.savefig('./figs/margin.svg', bbox_inches='tight')


def marginComparisonFigure(filename):
    pkl_filename = f"./corpus/{filename}.pkl"
    with open(pkl_filename, 'rb') as fp:
        result_tuple = pickle.load(fp)

    falseDiffList, trueDiffList, sameList = result_tuple

    diff_metric_list, same_metric_list = get_metric_list(trueDiffList, sameList, max_num=None)
    draw_diff(diff_metric_list, same_metric_list)


def differenceComparison(filename):
    pkl_filename = f"./corpus/{filename}.pkl"
    with open(pkl_filename, 'rb') as fp:
        result_tuple = pickle.load(fp)

    falseDiffList, trueDiffList, sameList = result_tuple

    diff_metric_list = []
    same_metric_list = []

    cnt_second = 0
    for data, label, pred_p, pred_e in trueDiffList:
        values, _ = torch.topk(pred_p, k=2)
        pln_diff = values[0][0] - values[0][1]

        values, _ = torch.topk(pred_e, k=2)
        enc_diff = values[0][0] - values[0][1]

        diff_metric_list.append(pln_diff.item() - enc_diff.item())

    for data, label, pred_p, pred_e in sameList:
        values, _ = torch.topk(pred_p, k=2)
        pln_diff = values[0][0] - values[0][1]

        values, _ = torch.topk(pred_e, k=2)
        enc_diff = values[0][0] - values[0][1]

        same_metric_list.append(pln_diff.item() - enc_diff.item())

    plt.rcParams["font.family"] = "Times New Roman"

    # plt.hist(diff_metric_list, bins=5, color='blue', alpha=0.3, label="Deviation")
    plt.hist(same_metric_list, bins=50, color='red', alpha=0.3, label="Normal")

    plt.title("Difference Value Comparison", fontsize=20)
    plt.yscale('log')
    plt.xlabel('Difference Value', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.yticks(size = 18)
    plt.xticks(size = 18)
    plt.legend(fontsize=18)
    plt.savefig('./figs/difference.svg', bbox_inches='tight')


if __name__ == "__main__":
    # Table 3
    # secondClassRatio("ori_ts_mnist")
    # secondClassRatio("ori_ts_digits")
    # secondClassRatio("ori_helayer_credit")
    # secondClassRatio("ori_helayers_bank")
    # secondClassRatio("ori_helayer_digits")
    # secondClassRatio("ori_helayer_mnist")

    # Figure 1
    # marginComparisonFigure("ori_helayer_credit")

    # Figure #
    differenceComparison("ori_helayer_mnist")