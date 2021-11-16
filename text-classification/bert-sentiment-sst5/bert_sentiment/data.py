"""This module defines a configurable SSTDataset class."""

import pytreebank
import torch
from loguru import logger
from pytorch_transformers import BertTokenizer
# from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np

logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

logger.info("Loading SST")
sst = pytreebank.load_sst()


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")


class SSTDataset(Dataset):
    """Configurable SST Dataset.
    
    Things we can configure:
        - split (train / val / test)
        - root / all nodes
        - binary / fine-grained
    """

    def __init__(self, split="train", root=True, binary=True, noise_type="clean", noise_rate = 0.5):
        """Initializes the dataset with given configuration.

        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """
        logger.info(f"Loading SST {split} set")
        self.sst = sst[split]

        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.nb_classes = 5

        logger.info("Tokenizing")
        if root and binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    get_binary_label(tree.label),
                )
                for tree in self.sst
                if tree.label != 2
            ]
        elif root and not binary:
            # 经常在这个部分
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]
        
        self.data_new = self.data.copy()
        if self.noise_type != "clean":
            logger.info("生成噪声")
            logger.info(len(self.data))
            # 生成噪声矩阵
            if self.noise_type == "symmetric":
                self.noisify_multiclass_symmetric()
            else:
                self.noisify_multiclass_pairflip()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.noise_type == "clean":
            X, y = self.data[index]
        else:  
            # print("here")
            X, y = self.data_new[index]
        X = torch.tensor(X)
        return X, y
    
    
    def noisify_multiclass_symmetric(self, random_state = None):
        P = np.ones((self.nb_classes, self.nb_classes)) # 噪声矩阵
        n = self.noise_rate
        P = (n / (self.nb_classes - 1)) * P
        flipper = np.random.RandomState(random_state)

        if n > 0.0:
            # 0 -> 1
            P[0, 0] = 1. - n
            for i in range(1, self.nb_classes-1):
                P[i, i] = 1. - n
            P[self.nb_classes-1, self.nb_classes-1] = 1. - n
        print(P)
        lenth  = len(self.data)
        y_train = []
        y_train_noise = []
        for i in np.arange(lenth):
            label = self.data[i][1]
            flipped = flipper.multinomial(1, P[label, :], 1)[0]
            # print(P[label, :], flipped, np.where(flipped == 1))
            self.data_new[i] = list(self.data_new[i])
            self.data_new[i][1] = np.where(flipped == 1)[0][0]
            self.data_new[i] = tuple(self.data_new[i])
            y_train.append(self.data[i][1])
            y_train_noise.append(self.data_new[i][1])
        
        y_train = np.array(y_train)
        y_train_noise = np.array(y_train_noise)
        actual_noise = (y_train != y_train_noise).mean()
        assert actual_noise > 0.0
        logger.info('Actual noise %.2f' % actual_noise)
    
    # pairfilp
    def noisify_multiclass_pairflip(self, random_state = None):
        P = P = np.eye(self.nb_classes) # 噪声矩阵
        n = self.noise_rate

        if n > 0.0:
            # 0 -> 1
            P[0, 0], P[0, 1] = 1. - n, n
            for i in range(1, self.nb_classes-1):
                P[i, i], P[i, i + 1] = 1. - n, n
            P[self.nb_classes-1, self.nb_classes-1], P[self.nb_classes-1, 0] = 1. - n, n
            lenth  = len(self.data)
        print(P)
        flipper = np.random.RandomState(random_state)

        y_train = []
        y_train_noise = []
        for i in np.arange(lenth):
            label = self.data[i][1]
            flipped = flipper.multinomial(1, P[label, :], 1)[0]
            # print(P[label, :], flipped, np.where(flipped == 1))
            self.data_new[i] = list(self.data_new[i])
            self.data_new[i][1] = np.where(flipped == 1)[0][0]
            self.data_new[i] = tuple(self.data_new[i])
            y_train.append(self.data[i][1])
            y_train_noise.append(self.data_new[i][1])
        
        y_train = np.array(y_train)
        y_train_noise = np.array(y_train_noise)
        actual_noise = (y_train != y_train_noise).mean()
        assert actual_noise > 0.0
        logger.info('Actual noise %.2f' % actual_noise)
        
        
        