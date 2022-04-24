from config import *
from torch.utils.data import DataLoader, Dataset
import os, random
from collections import Counter
import nltk
import numpy as np
import torch
import torch.nn.functional as F


rand0=np.array(
      [ 0.45516681, -0.07138701,  0.82986405, -1.11720541, -0.33429333,
        0.69099271, -1.85049927,  0.78906409,  0.89888104, -1.67096037,
        0.25958214, -0.03233769, -0.62160056, -1.39370215, -1.14012322,
        1.85143923,  0.26782411,  0.39617544,  2.17566168, -0.97191126,
       -0.4568121 ,  0.1717204 , -0.74570957, -0.76992489,  0.57583092,
       -0.76224013,  1.54899431, -0.58142558, -0.20596733,  0.63618401,
       -1.36106748,  0.42459613,  0.51972894, -0.76357479,  0.14653988,
        0.0581039 ,  0.37996582, -0.45868945,  0.26124548, -0.17687444,
        0.49617259, -0.3576333 ,  0.11327246, -1.14447653, -0.78943423,
        1.03538124, -1.15449178, -0.13961598, -0.18839142, -0.74496439,
        0.40360879, -0.60476766, -0.86831956, -0.44928197,  1.84981372,
        0.74881436, -0.42642061, -1.41438152, -0.89824092,  0.94428695,
       -0.91881174, -0.2778995 , -1.17882606, -1.51356166, -2.05036475,
       -1.06149148, -0.04956149, -0.61472062,  0.30535658,  0.79543112,
       -1.48068996, -0.61028809,  0.32866681, -1.01110574, -2.18402496,
       -0.88692738, -1.66619197, -0.13239246, -0.20970241,  0.7912142 ,
       -1.0229638 , -2.05980907, -1.45587505,  2.80367757, -1.06088623,
        0.55481641,  1.05536402,  1.77283806,  0.52583477,  1.15572462,
        1.04665724, -0.75035667, -1.54611217,  0.81360407,  0.20823966,
        1.4719338 ,  0.60902381,  0.11183052,  1.37372793,  0.90768606]
    )

# nltk.download('punkt')

class IMDBMovieReviews():
    def __init__(self, root):
        self.root = root
        return

    def get_data(self):
        def extract_data(dir, split):
            data = []
            for label in ("pos", "neg"):
                label_dir = os.path.join(dir, "aclImdb", split, label)
                files = sorted(os.listdir(label_dir))
                for file in files:
                    filepath = os.path.join(label_dir, file)
                    with open(filepath, encoding="UTF-8") as f:
                        data.append({L_RAW: f.read(), L_LABEL: label})
            return data

        train_data = extract_data(self.root, "train")
        test_data = extract_data(self.root, "test")
        return train_data, test_data

    def split_data(self, train_data, num_split=2000):
        random.seed(50)
        random.shuffle(train_data, random=lambda : 0.5)
        return train_data[:-num_split], train_data[-num_split:]

    def tokenize(self, data, max_seq_len=MAX_SEQ_LEN):
        for review in data:
            review[L_TOKENS] = []
            for sent in nltk.sent_tokenize(review[L_RAW]):
                review[L_TOKENS].extend(nltk.word_tokenize(sent))
            if max_seq_len >= 0:
                review[L_TOKENS] = review[L_TOKENS][:max_seq_len]

    def create_vocab(self, data, unk_threshold=UNK_THRESHOLD):
        counter = Counter(token for review in data for token in review[L_TOKENS])
        self.vocab = {token for token in counter if counter[token] > unk_threshold}
        token_to_idx = {PAD: 0, UNK: 1}
        for token in self.vocab:
            token_to_idx[token] = len(token_to_idx)
        return token_to_idx

    def get_embeds(self, token_to_index_mapping, token_to_glove, dim):
        weights_matrix = np.zeros((len(token_to_index_mapping), dim))
        indices_found = []

        for word, i in token_to_index_mapping.items():
            if word in token_to_glove.keys():
                indices_found.append(i) # This gradient of these indices will get zero'd out later depending on config
            weights_matrix[i] = token_to_glove.get(word, np.random.RandomState(i).normal(size=(100, ))) # np.random.normal(size=(dim, ))
        return indices_found, weights_matrix

    def apply_vocab(self, data, token_to_idx):
        for review in data:
            review[L_TOKENS] = [token_to_idx.get(token, token_to_idx[UNK]) for token in review[L_TOKENS]]


    def apply_label_map(self, data, label_to_idx):
        for review in data:
            review[L_LABEL] = label_to_idx[review[L_LABEL]]

class SentimentDataset(Dataset):
    def __init__(self, data, pad_idx):
        data = sorted(data, key=lambda review: len(review[L_TOKENS]))
        self.texts = [review[L_TOKENS] for review in data]
        self.labels = [review[L_LABEL] for review in data]
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index]]

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        def tensorize(elements, dtype):
            return [torch.tensor(element, dtype=dtype) for element in elements]

        def pad(tensors):
            max_len = max(len(tensor) for tensor in tensors)
            padded_tensors = [
                F.pad(tensor, (0, max_len - len(tensor)), value=self.pad_idx) for tensor in tensors
            ]
            return padded_tensors

        texts, labels = zip(*batch)
        return [
            torch.stack(pad(tensorize(texts, torch.long)), dim=0),
            torch.stack(tensorize(labels, torch.long), dim=0),
        ]