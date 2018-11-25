import math


def creat_data():
    datasets = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return datasets, labels


def shannon_ent(datasets, labels):
    num_entries = len(datasets)
    label_count = {}
    for entry in datasets:
        # 获取没一条数据的标签
        current_feature = entry[-1]
        if current_feature not in label_count.keys():
            label_count[current_feature] = 0
        label_count[current_feature] += 1

    shannon_ent = 0.0
    # 求香农信息熵
    for key in label_count:
        prob = float(label_count[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)

    return shannon_ent

