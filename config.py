import torch

class Config():
    def __init__(self):
        self.data_path = 'data/en-cn.txt'
        self.word_count = 10000 #根据实际需要预测的值改变
        self.model = 512
        self.layers = 6
        self.heads = 8
        self.d_q = 64
        self.d_k = 64
        self.d_v = 64
        self.d_ff = 1024
        self.pro_word_count = 100
        self.epochs = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')