from utils import normalizeString, cht_to_chs
from config import Config
import torch
from torch.autograd import Variable
import random

'''读取数据原始标注的数据'''
config = Config()


def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        english_data, chinese_data, chinese_data_eval = [], [], []
        for line in f.readlines():
            english_data.append(normalizeString(line.split('\t')[0]))
            chinese_data.append('S' + cht_to_chs(line.split('\t')[1]))
            chinese_data_eval.append(cht_to_chs(line.split('\t')[1] + 'E'))

        return english_data, chinese_data, chinese_data_eval


def data_list(data_path):
    english_data, chinese_data, chinese_data_eval = read_data(data_path)
    english_data_list = []
    for line in english_data:
        for word in line.split():
            english_data_list.append(word)
    chinese_data_list = []
    for line in chinese_data:
        for word in line:
            chinese_data_list.append(word)
    chinese_data_eval_list = []
    for line in chinese_data_eval:
        for word in line:
            chinese_data_eval_list.append(word)
    return english_data_list, chinese_data_list, chinese_data_eval_list


class Data():
    def __init__(self, name):
        self.name = name
        self.word_index = {'S': 2, 'E': 1, 'P': 0}
        self.word_count = {'S': 2, 'E': 1, 'P': 0}
        self.index_word = {2: "S", 1: "E", 0: 'P'}
        self.n_word = 2

    def process(self, data):
        for word in data:
            if word not in self.word_index.keys():
                self.word_index[word] = self.n_word
                self.word_count[word] = self.n_word
                self.index_word[self.n_word] = word
                self.n_word += 1
            else:
                self.word_count[word] += 1


def pro_data(data_path):
    data = []
    english_data = Data('english')
    chinese_data = Data('chinese')
    print(english_data, chinese_data)
    # chinese_eval = Data('chinese_eval')
    english_data_list1, chinese_data_list1, chinese_data_eval_list1 = read_data(data_path)
    english_data_list, chinese_data_list, chinese_data_eval_list = data_list(data_path)

    english_data.process(english_data_list)
    chinese_data.process(chinese_data_list)

    for english, chinese, chinese_eval in zip(english_data_list1, chinese_data_list1, chinese_data_eval_list1):
        english_tensor = Variable(torch.LongTensor([english_data.word_index[word] for word in english.split()]))
        chinese_tensor = Variable(torch.LongTensor([chinese_data.word_index[word] for word in chinese]))
        chinese_eval_tensor = Variable(torch.LongTensor([chinese_data.word_index[word] for word in chinese_eval]))
        data.append([english_tensor, chinese_tensor, chinese_eval_tensor])

    return data


# 自定义的Dataset函数的，对数切分，不需要对数据进行补全成长度一样的数据
def Dataset(data1, batch_size, batch_num):
    data = data1
    random.shuffle(data)
    train_data1 = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    eval_data = data[int(len(data) * 0.9):]
    train_data = []
    for _ in range(batch_num):
        train_data.append(random.sample(train_data1, batch_size))

    return train_data, test_data, eval_data


if __name__ == '__main__':
    config = Config()
    data = pro_data(config.data_path)
    for line in data:
        print(line)
    print('___________________')
    a, b, c = Dataset(data, batch_size=3, batch_num=5)
    for line in a:
        print(line)
