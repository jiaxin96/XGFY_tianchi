import argparse
import csv
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import random
import pandas as pd
import numpy as np
import sys
import re

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

def augment_data(source_path='train.csv', target_path='train_aug.csv'):
    root = './data/input/'
    df = pd.read_csv(os.path.join(root, source_path))
    df = df.fillna(0)
    data = df.values.tolist()
    data_ = []
    for d in data:
        d_ = d[1:] #去除类别
        data_.append(d_)
    data_add = []
    for i in range(len(data_)):
        for j in range(i+1, i+32): #据观察没有超过连着20条连续一样的query1
            if j < len(data_): #防越界
                if data_[i][0] == data_[j][0]: #若为相同的query1
                    if data_[i][2]==1 and data_[j][2]==1: #若label均为1
                        add_item = [data_[i][1], data_[j][1], int(1)]
                        data_add.append(add_item)
                    elif data_[i][2]==1 or data_[j][2]==1: #若一个为0一个为1
                        add_item = [data_[i][1], data_[j][1], int(0)]
                        data_add.append(add_item)
    data_.extend(data_add)
    target = pd.DataFrame(data=data_, columns=["query1","query2","label"])
    target['label'] = target['label'].astype(int)
    target.to_csv(os.path.join(root, target_path), index=False)

def clean_data(source_path='dev.csv', target_path='dev_clean.csv'):
    root = './data/input/'
    df = pd.read_csv(os.path.join(root, source_path))
    df = df.fillna(0)
    data = df.values.tolist()
    data_ = []
    for d in data:
        d_ = d[1:] #去除类别
        data_.append(d_)
    target = pd.DataFrame(data=data_, columns=["query1","query2","label"])
    target['label'] = target['label'].astype(int)
    target.to_csv(os.path.join(root, target_path), index=False)







class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


#格式化数据并读取
class SimProcessor(DataProcessor):
    @classmethod
    def get_train_examples(cls, data_dir='./data/input/', data_name='train_aug.csv'):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "input")))

        file_path = os.path.join(data_dir, data_name)
        train_df = pd.read_csv(file_path, encoding='utf-8', header=0)
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            text_a = str(train[0])
            text_b = str(train[1])
            label = str(train[2])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return train_data

    @classmethod
    def get_dev_examples(cls, data_dir='./data/input/', data_name='dev_clean.csv'):
        file_path = os.path.join(data_dir, data_name)
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = str(dev[0])
            text_b = str(dev[1])
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data
        #序号、sen1、sen2、类别

    #返回所有的类别
    @classmethod
    def get_labels(cls):
        """See base class."""
        return ["0", "1"]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # 截断文本
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

if __name__ == '__main__':
    # augment_data()
    clean_data()
    



