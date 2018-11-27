import math
import numpy as np


def creat_data():
    datasets = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return datasets, labels


def shannon_ent(datasets):
    """
    计算数据集的信息熵
    :param datasets:
    :param labels:
    :return:
    """
    num_entries = len(datasets)
    label_count = {}
    for entry in datasets:
        # 获取每一条数据的标签
        current_label = entry[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1

    shannon_ent = 0.0
    # 求香农信息熵
    for key in label_count:
        prob = float(label_count[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)

    return shannon_ent


def split_dataset(dataset, index, value):
    """
    :param dataset: 待划分的数据集
    :param index: index对应的列
    :param value: index列对应的value值
    :return res_data: 分割后的数据集
    """
    res_data = []
    for ele in dataset:
        if ele[index] == value:
            # 获取不包含index列的数据
            reduced_dataset = ele[:index]
            reduced_dataset.extend(ele[index+1:])
            res_data.append(reduced_dataset)
    return res_data


def best_feature_split(dataset):
    """
    :param dataset:
    :return:

    """
    # 获取feature的数量
    features = len(dataset[0]) - 1

    best_info_gain, best_feature = 0.0, -1
    base_entroy = shannon_ent(dataset)

    for i in range(features):
        # 遍历特征,获取该特征的所有数据
        feat_values= [ele[i] for ele in dataset]
        print(feat_values)
        # 去重
        uniq_feat = set(feat_values)
        # 子数据集的信息熵
        new_entropy = 0.0
        for feature in feat_values:
            # 获取子数据集
            sub_data = split_dataset(dataset, i, feature)
            pro = len(sub_data) / float(len(dataset))
            new_entropy += pro * shannon_ent(sub_data)

        # 信息增益:原数据集的信息熵 - 分割后的数据集的信息熵
        info_gain = base_entroy - new_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
            print(best_info_gain, best_feature)
    return best_feature


if __name__ == '__main__':
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # dataset = np.array(dataSet)
    dataset, labels = creat_data()
    ent = shannon_ent(dataset)
    print(best_feature_split(dataset))
