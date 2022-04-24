import os
import torch
from torch import optim
from torch.nn import RNN, LSTM, LSTMCell
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional as F
import random


def load_data(path, flag='train'):
    labels = ['pos', 'neg']
    data = []
    for label in labels:
        files = os.listdir(os.path.join(path, flag, label))
        # 去除标点符号
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
        for file in files:
            with open(os.path.join(path, flag, label, file), 'r', encoding='utf8') as rf:
                temp = rf.read().replace('\n', '')
                temp = temp.replace('<br /><br />', ' ')
                temp = re.sub(r, '', temp)
                temp = temp.split(' ')
                temp = [temp[i].lower() for i in range(len(temp)) if temp[i] != '']
                if label == 'pos':
                    data.append([temp, 1])
                elif label == 'neg':
                    data.append([temp, 0])
    return data


# 对每一个句子进行处理，最大长度为250
def process_sentence():
    sentence_code = []
    vocabulary_vectors = np.load('vocabulary_vectors_1.npy', allow_pickle=True)
    word_list = np.load('word_list_1.npy', allow_pickle=True)
    word_list = word_list.tolist()

    # train_data = load_data('../\.data/imdb/aclImdb', 'train')
    # print("process_sentence train!!")
    # for i in range(len(train_data)):
    #     if i%100==0: print(i)
    #     vec = train_data[i][0]
    #     temp = []
    #     index = 0
    #     for j in range(len(vec)):
    #         try:
    #             index = word_list.index(vec[j])
    #         except ValueError:  # 没找到
    #             index = 399999
    #         finally:
    #             temp.append(index)  # temp表示一个单词在词典中的序号
    #     if len(temp) < 250:
    #         for k in range(len(temp), 250):  # 不足补0
    #             temp.append(0)
    #     else:
    #         temp = temp[0:250]  # 只保留250个
    #     sentence_code.append(temp)

    # # print(sentence_code)

    # sentence_code = np.array(sentence_code)
    # np.save('sentence_code_1', sentence_code)  # 存下来

    sentence_code = []
    test_data = load_data('../\.data/imdb/aclImdb', 'test')
    print("process_sentence test!!")
    for i in range(len(test_data)):
        if i%100==0: print(i)
        vec = test_data[i][0]
        temp = []
        index = 0
        for j in range(len(vec)):
            try:
                index = word_list.index(vec[j])
            except ValueError:  # 没找到
                index = 399999
            finally:
                temp.append(index)  # temp表示一个单词在词典中的序号
        if len(temp) < 250:
            for k in range(len(temp), 250):  # 不足补0
                temp.append(0)
        else:
            temp = temp[0:250]  # 只保留250个
        sentence_code.append(temp)

    # print(sentence_code)

    sentence_code = np.array(sentence_code)
    np.save('sentence_code_2', sentence_code)  # 存下来


# 定义词向量表
def load_cab_vector():
    word_list = []
    vocabulary_vectors = []
    data = open("../\.vector_cache/glove.6B.100d.txt", encoding='utf-8')
    for line in data.readlines():
        temp = line.strip('\n').split(' ')  # 一个列表
        name = temp[0]
        word_list.append(name.lower())
        vector = [temp[i] for i in range(1, len(temp))]  # 向量
        vector = list(map(float, vector))  # 变成浮点数
        vocabulary_vectors.append(vector)
    # 保存
    vocabulary_vectors = np.array(vocabulary_vectors)
    word_list = np.array(word_list)
    np.save('vocabulary_vectors_1', vocabulary_vectors)
    np.save('word_list_1', word_list)
    return vocabulary_vectors, word_list

# 分批处理数据
def process_batch(batchSize):
    # process_sentence()
    index = [i for i in range(25000)]
    random.shuffle(index)
    # 25000维的训练集与数据集
    test_data = load_data('../\.data/imdb/aclImdb', flag='test')
    train_data = load_data('../\.data/imdb/aclImdb')
    # shuffle
    train_data = [train_data[i] for i in index]
    test_data = [test_data[i] for i in index]
    # 加载句子的索引
    sentence_code_1 = np.load('sentence_code_1.npy', allow_pickle=True)
    sentence_code_1 = sentence_code_1.tolist()
    sentence_code_1 = [sentence_code_1[i] for i in index]
    # 25000 * 250测试集
    sentence_code_2 = np.load('sentence_code_2.npy', allow_pickle=True)
    sentence_code_2 = sentence_code_2.tolist()
    sentence_code_2 = [sentence_code_2[i] for i in index]
    vocabulary_vectors = np.load('vocabulary_vectors_1.npy', allow_pickle=True)
    vocabulary_vectors = vocabulary_vectors.tolist()

    # 每个sentence_code都是25000 * 250 * 50
    for i in range(25000):
        for j in range(250):
            sentence_code_1[i][j] = vocabulary_vectors[sentence_code_1[i][j]]
            sentence_code_2[i][j] = vocabulary_vectors[sentence_code_2[i][j]]
    labels_train = []
    labels_test = []
    arr_train = []
    arr_test = []

    # mini-batch操作
    for i in range(1, 251):
        arr_train.append(sentence_code_1[(i - 1) * batchSize:i * batchSize])
        labels_train.append([train_data[j][1] for j in range((i - 1) * batchSize, i * batchSize)])
        arr_test.append(sentence_code_2[(i - 1) * batchSize:i * batchSize])
        labels_test.append([test_data[j][1] for j in range((i - 1) * batchSize, i * batchSize)])

    arr_train = np.array(arr_train)
    arr_test = np.array(arr_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    # np.save('arr_train', arr_train)
    # np.save('arr_test', arr_test)
    # np.save('labels_train', labels_train)
    # np.save('labels_test', labels_test)

    return arr_train, labels_train, arr_test, labels_test


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        x = input
        x, (h_n, c_n) = self.lstm(x)
        output_f = h_n[-2, :, :]
        output_b = h_n[-1, :, :]
        output = torch.cat([output_f, output_b], dim=-1)
        out_fc1 = self.fc1(output)
        out_relu = F.relu(out_fc1)
        out = self.fc2(out_relu)
        # 概率
        return F.log_softmax(out, dim=-1)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.RNN(input_size=100, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        x = input
        x, (h_n, c_n) = self.lstm(x)
        output_f = h_n[-2, :, :]
        output_b = h_n[-1, :, :]
        output = torch.cat([output_f, output_b], dim=-1)
        out_fc1 = self.fc1(output)
        out_relu = F.relu(out_fc1)
        out = self.fc2(out_relu)
        # 概率
        return F.log_softmax(out, dim=-1)



# 训练与测试
def main():
    # load_cab_vector()
    # 加载各种数据
    print('loading...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    epoch_num = 50
    # arr_train为250 * 100 * 250 * 50
    arr_train, labels_train, arr_test, labels_test = process_batch(100)

    print('training...')
    net = RNN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    print(len(arr_train))
    print(len(arr_test))
    for i in range(epoch_num):
        print("epoch:",end="")
        num=0
        for j in range(250):
            x = arr_train[j]
            y = labels_train[j]
            input_ = torch.tensor(x, dtype=torch.float32).to(device)
            label = torch.tensor(y, dtype=torch.long).to(device)
            output = net(input_)
            pred = output.max(dim=-1)[1]
            for k in range(100):
                if pred[k] == label[k]:
                    num += 1
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, label)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        print('%d loss:%.5f' % (i, loss.item()))
        print('Accuracy：', num / 25000)
        print('testing...')
        num = 0
        for j in range(250):
            xx = arr_test[j]
            yy = labels_test[j]
            input_ = torch.tensor(xx, dtype=torch.float32).to(device)
            label = torch.tensor(yy, dtype=torch.long).to(device)
            output = net(input_)
            pred = output.max(dim=-1)[1]
            for k in range(100):
                if pred[k] == label[k]:
                    num += 1
        print('Accuracy：', num / 25000)


if __name__ == '__main__':
    main()

