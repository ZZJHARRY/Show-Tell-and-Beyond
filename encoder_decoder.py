import pandas as pd
import numpy as np
from collections import Counter
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import pickle
import gc
import random
import cv2
from google.colab.patches import cv2_imshow
import argparse, os, sys
from sklearn.metrics import accuracy_score
from torchvision import transforms

# Hyperparameters
HIDDEN_DIM = 200
BATCH_SIZE = 10
LEARN_RATE = 0.001
NUM_EPOCHS = 10

def extract_captions(fn):
    image_captions_train, image_captions_test = [], []
    with open(fn) as f:
        for line in f.readlines():
            arr = line.split()
            arr = arr[0].split(',') + ['<s>'] + arr[1:] + ['</s>']
            if arr[0] in features_train:
                image_captions_train.append(arr)
            if arr[0] in features_test:
                image_captions_test.append(arr)
    return image_captions_train, image_captions_test

def extract_dict(fn):
    index_to_word, word_to_index = ['PAD'], {'PAD':0}
    image_captions = image_captions_train + image_captions_test
    for image_caption in image_captions:
        for word in image_caption[1:]:
            if word not in index_to_word:
                word_to_index[word] = len(index_to_word)
                index_to_word.append(word)
    return index_to_word, word_to_index, len(index_to_word)

def caption_to_tensor(caption, max_len_batch):
    caption_arr = np.zeros(max_len_batch)
    caption += ['PAD' for i in range(max_len_batch-len(caption))]
    for i in range(max_len_batch):
        caption_arr[i] = word_to_index[caption[i]]
    return torch.Tensor(caption_arr)

def scores_to_caption(scores):
    predicted_indices = torch.argmax(scores, -1)
    predicted_captions = []
    for i in range(n_test):
        j = 0
        while j
    return ret
    
def make_batch(batch_number):
    batch_index_first = batch_number * BATCH_SIZE
    batch_index_final = min(n_train, (batch_number+1)*BATCH_SIZE)
    # compute maximum caption length in batch
    max_len_batch = 0
    for i in range(batch_index_first, batch_index_final):
        if len(image_captions_train[i][1:]) > max_len_batch:
            max_len_batch = len(image_captions_train[i][1:])
    X_train_batch = torch.zeros((BATCH_SIZE, feature_dim[0],feature_dim[1], feature_dim[2]))
    Y_train_batch = torch.zeros((BATCH_SIZE, max_len_batch))
    for i in range(batch_index_first, batch_index_final):
        j = i - batch_index_first
        image, caption = image_captions_train[i][0], image_captions_train[i][1:]
        X_train_batch[j] = features_train[image]
        Y_train_batch[j] = caption_to_tensor(caption, max_len_batch)
    return X_train_batch, torch.flatten(Y_train_batch), Y_train_batch, max_len_batch

def make_test():
    X_test = torch.zeros((n_test, feature_dim[0],feature_dim[1], feature_dim[2]))
    for i in range(0, n_test):
        image = image_captions_test[i][0]
        X_test[i] = features_test[image]
    return X_test

class EncoderCNN(nn.Module):

    def __init__(self, feature_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=feature_dim[0], out_channels=256, kernel_size=(3,3), stride=1, padding=0),
                                     torch.nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=1, padding=0),
                                     torch.nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
                                     nn.ReLU(),
                                     nn.Linear(in_features=128*feature_dim[1]*feature_dim[2], out_features=embedding_dim))

    def forward(self, features):
        return self.encoder(features)

class DecoderLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(RNNTagger, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, in_sequence):
        lstm_out, _ = self.lstm(in_sequence)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.softmax(tag_space)
        return tag_scores

def teacher_forcer(encoded_features, Y_train_batch_unflattened, max_len_batch):
    label_output = np.zeros((BATCH_SIZE, max_len_batch, embedding_dim))
    # Index of samples in the current batch
    for idx_sentence in range(BATCH_SIZE):
        for idx_word in range(max_len_batch):
            if idx_word == 0:
                label_output[idx_sentence, idx_word, :] = encoded_features[idx_sentence, :]
            else:
                label_output[idx_sentence, idx_word, Y_train_batch_unflattened[idx_sentence, idx_word - 1]] = 1
    return label_output

image_captions_train, image_captions_test = extract_captions('captions.txt')
n_train, n_test = len(image_captions_train), len(image_captions_test)
    

pd.set_option('display.max_colwidth', None)

features_train, features_test = extract_imgFtr_ResNet_train, extract_imgFtr_ResNet_valid
n_train_images, n_test_images = len(features_train), len(features_test)
feature_dim = features_train['436015762_8d0bae90c3.jpg'][0].shape

index_to_word, word_to_index, embedding_dim = extract_dict('captions.txt')

X_train_batch, Y_train_batch, Y_train_batch_unflattened, max_len_batch = make_batch(0)

X_test = make_test()
