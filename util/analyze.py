import unittest
import os
import os.path
import ast

import numpy as np
import pandas as pd
import logging
from ..loader.official import OfficialLoader
import torch
import torch.nn as nn
import os
import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

os.environ['PIP_CACHE_DIR'] = "/vol/bitbucket/ya321/.cache"
os.environ['TRANSFORMERS_CACHE'] = "/vol/bitbucket/ya321/.cache"
os.environ['HF_DATASETS_CACHE'] = "/vol/bitbucket/ya321/.cache"
os.environ["WANDB_DISABLED"] = "true"


def get_predicted_labels(probs):
    softMax = nn.Softmax(1)
    logits = torch.tensor(probs)
    output = softMax(logits)
    labels = np.argmax(output.detach().cpu().numpy(), axis=-1)
    print(f'labels.shape = {labels.shape}')
    return labels


class DataAnalyseTestCase(unittest.TestCase):

    def test_gain_data_with_oriinal_label(self):
        loader = OfficialLoader('Analysis')
        loader.generate_data_original_labels()

    @staticmethod
    def test_all_label():
        loader = OfficialLoader('AnalysisP')
        loader.split()
        predicted_probs_path = os.path.join(loader.prob_dir,
                                            loader.prob_filename)
        predict_probs = np.loadtxt(predicted_probs_path)
        predict_labels = get_predicted_labels(predict_probs)
        loader.eval(np.array(predict_labels), np.array(loader.test_data['label']))

    def test_analyze_result_per_level(self):
        loader = OfficialLoader('AnalysisPerLevel')
        if os.path.isfile(os.path.join(loader.data_dir, loader.official_test_data_five_labels)):
            test_data_all_labels = pd.read_csv(os.path.join(loader.data_dir, loader.official_test_data_five_labels),
                                               usecols=['par_id', 'text', 'binary_label', 'orig_label', 'category'])
            predicted_probs_path = os.path.join(loader.prob_dir,
                                                loader.prob_filename)
            predict_probs = np.loadtxt(predicted_probs_path)
            predict_labels = get_predicted_labels(predict_probs)
            print(f"predict_label.shape = {predict_labels.shape}")
            test_data_all_labels.loc[:, 'predicted_label'] = predict_labels[:]
            print(f'test_data_all_labels.shape = {test_data_all_labels.shape}')
            print(f'test_data_all_labels.head = \n {test_data_all_labels.head()}')
            # level_dict[level] [0] for true label, [1] for predicted label
            level_dict = {i: [[], []] for i in range(5)}
            level_category_count = {i: [0 for j in range(7)] for i in [2, 3, 4]}
            for i, data in tqdm.tqdm(test_data_all_labels.iterrows(), total=test_data_all_labels.shape[0]):
                # print(data)
                level_dict[data['orig_label']][0].append(int(data['binary_label']))
                level_dict[data['orig_label']][1].append(int(data['predicted_label']))
                cate = ast.literal_eval(data['category'])
                if isinstance(cate, int):
                    if cate == -1:
                        continue
                    cate = [cate]
                level_category_count[data['orig_label']][len(cate) - 1] += 1
            # print(level_dict)
            for key, data in level_dict.items():
                loader.eval_per(np.array(data[0]), np.array(data[1]), 'level', key)
            for key, val in level_category_count.items():
                total_label_num = 0
                for num, count in enumerate(val):
                    total_label_num += (num + 1) * count
                print(f'level{key}, val = {val}\tcategory_count_avg = {total_label_num / np.sum(val)}')
                plt.cla()
                plt.bar(range(1, 8), val)
                plt.title(f'level{key} label_count')
                plt_path = './prob/analysis_plot/'
                if not os.path.exists(plt_path):
                    os.makedirs(plt_path)
                plt.savefig(plt_path + f'level{key}_label_count')
        else:
            print(f'{os.path.join(loader.data_dir, loader.official_test_data_five_labels)} doesn\'t exists')

    def test_analyze_per_length(self):
        loader = OfficialLoader('AnalysisPerLength')
        if os.path.isfile(os.path.join(loader.data_dir, loader.official_test_data_five_labels)):
            test_data_all_labels = pd.read_csv(os.path.join(loader.data_dir, loader.official_test_data_five_labels),
                                               usecols=['par_id', 'text', 'binary_label', 'orig_label', 'category'])
            predicted_probs_path = os.path.join(loader.prob_dir,
                                                loader.prob_filename)
            predict_probs = np.loadtxt(predicted_probs_path)
            predict_labels = get_predicted_labels(predict_probs)
            print(f"predict_label.shape = {predict_labels.shape}")
            test_data_all_labels.loc[:, 'predicted_label'] = predict_labels[:]
            print(f'test_data_all_labels.shape = {test_data_all_labels.shape}')
            print(f'test_data_all_labels.head = \n {test_data_all_labels.head()}')
            length_dict = {i: [[], []] for i in range(4)}
            key_to_len_dict = {0: '0-127', 1: '128-255', 2: '255-511', 3: '512+'}
            for i, data in tqdm.tqdm(test_data_all_labels.iterrows(), total=test_data_all_labels.shape[0]):
                # print(data)
                length = len(data['text'])
                if length < 128:
                    # 0: 0 - 256
                    length_dict[0][0].append(int(data['binary_label']))
                    length_dict[0][1].append(int(data['predicted_label']))
                elif length < 256:
                    # 1: 256 - 512
                    length_dict[1][0].append(int(data['binary_label']))
                    length_dict[1][1].append(int(data['predicted_label']))
                elif length < 512:
                    # 2: 512 - 1024
                    length_dict[2][0].append(int(data['binary_label']))
                    length_dict[2][1].append(int(data['predicted_label']))
                else:
                    # 3: 1024+
                    length_dict[3][0].append(int(data['binary_label']))
                    length_dict[3][1].append(int(data['predicted_label']))
            for key, data in length_dict.items():
                loader.eval_per(np.array(data[0]), np.array(data[1]), 'length', key_to_len_dict[key])
        else:
            print(f'{os.path.join(loader.data_dir, loader.official_test_data_five_labels)} doesn\'t exists')

    def test_analyze_per_category(self):
        loader = OfficialLoader('AnalysisPerCategory')
        if os.path.isfile(os.path.join(loader.data_dir, loader.official_test_data_five_labels)):
            test_data_all_labels = pd.read_csv(os.path.join(loader.data_dir, loader.official_test_data_five_labels),
                                               usecols=['par_id', 'text', 'binary_label', 'orig_label', 'category'])
            predicted_probs_path = os.path.join(loader.prob_dir,
                                                loader.prob_filename)
            predict_probs = np.loadtxt(predicted_probs_path)
            predict_labels = get_predicted_labels(predict_probs)
            print(f"predict_label.shape = {predict_labels.shape}")
            test_data_all_labels.loc[:, 'predicted_label'] = predict_labels[:]
            print(f'test_data_all_labels.shape = {test_data_all_labels.shape}')
            print(f'test_data_all_labels.head = \n {test_data_all_labels.head()}')
            category_dict = {i: [[], []] for i in range(7)}
            for i, data in tqdm.tqdm(test_data_all_labels.iterrows(), total=test_data_all_labels.shape[0]):
                cate = ast.literal_eval(data['category'])
                if isinstance(cate, int):
                    cate = [cate]
                for c in cate:
                    if c < 0:
                        # non pcl
                        continue
                    category_dict[c][0].append(int(data['binary_label']))
                    category_dict[c][1].append(int(data['predicted_label']))
            for key, data in category_dict.items():
                loader.eval_per(np.array(data[0]), np.array(data[1]), 'category', key)
        else:
            print(f'{os.path.join(loader.data_dir, loader.official_test_data_five_labels)} doesn\'t exists')

    def test_analyze_per_country_keyword(self):
        loader = OfficialLoader('AnalysisPerCountryKeyword')
        if os.path.isfile(os.path.join(loader.data_dir, loader.official_test_data_five_labels)):
            test_data_all_labels = pd.read_csv(os.path.join(loader.data_dir, loader.official_test_data_five_labels),
                                               usecols=['par_id', 'text', 'binary_label',
                                                        'keyword', 'country'])
            predicted_probs_path = os.path.join(loader.prob_dir,
                                                loader.prob_filename)
            predict_probs = np.loadtxt(predicted_probs_path)
            predict_labels = get_predicted_labels(predict_probs)
            test_data_all_labels.loc[:, 'predicted_label'] = predict_labels[:]
            country_dict = {i: [[], []] for i in test_data_all_labels['country'].unique()}
            keyword_dict = {i: [[], []] for i in test_data_all_labels['keyword'].unique()}
            for i, data in tqdm.tqdm(test_data_all_labels.iterrows(), total=test_data_all_labels.shape[0]):
                country_dict[data['country']][0].append(int(data['binary_label']))
                country_dict[data['country']][1].append(int(data['predicted_label']))
                keyword_dict[data['keyword']][0].append(int(data['binary_label']))
                keyword_dict[data['keyword']][1].append(int(data['predicted_label']))
            for key, data in country_dict.items():
                loader.eval_per(np.array(data[0]), np.array(data[1]), 'country', key)
            for key, data in keyword_dict.items():
                loader.eval_per(np.array(data[0]), np.array(data[1]), 'keyword', key)
        else:
            print(f'{os.path.join(loader.data_dir, loader.official_test_data_five_labels)} doesn\'t exists')

    def test_analyze_train_data_per_level(self):
        loader = OfficialLoader('AnalysisTrainPerLevel')
        if os.path.isfile(os.path.join(loader.data_dir, loader.official_train_data_five_labels)):
            train_data_all_labels = pd.read_csv(os.path.join(loader.data_dir, loader.official_train_data_five_labels),
                                                usecols=['par_id', 'text', 'binary_label', 'orig_label', 'category'])
            level_dict = {i: [[], []] for i in range(5)}
            level_category_count = {i: [0 for j in range(7)] for i in [2, 3, 4]}
            for i, data in tqdm.tqdm(train_data_all_labels.iterrows(), total=train_data_all_labels.shape[0]):
                # print(data)
                level_dict[data['orig_label']][0].append(int(data['binary_label']))
                cate = ast.literal_eval(data['category'])
                if isinstance(cate, int):
                    if cate == -1:
                        continue
                    cate = [cate]
                level_category_count[data['orig_label']][len(cate) - 1] += 1
            # print(level_dict)
            for key, val in level_category_count.items():
                total_label_num = 0
                for num, count in enumerate(val):
                    total_label_num += (num + 1) * count
                print(f'level{key}, val = {val}\tcategory_count_avg = {total_label_num / np.sum(val)}')
                plt.cla()
                plt.bar(range(1, 8), val)
                plt.title(f'level{key} label_count')
                plt_path = './prob/analysis_train_plot/'
                if not os.path.exists(plt_path):
                    os.makedirs(plt_path)
                plt.savefig(plt_path + f'level{key}_label_count')

        else:
            print(f'{os.path.join(loader.data_dir, loader.official_test_data_five_labels)} doesn\'t exists')
