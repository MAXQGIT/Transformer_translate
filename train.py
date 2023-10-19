from data_process import pro_data, Dataset, Data, data_list
from config import Config
from torch import optim
import torch.nn as nn
import torch
from transformer import Transformers
import os

config = Config()

# 将英文和汉语对应的字典保存
if os.path.exists('data/{}.txt'.format('english_dict')):
    os.remove('data/{}.txt'.format('english_dict'))

with open('data/{}.txt'.format('english_dict'), 'a', encoding='utf-8') as f:
    english_data = Data('english')
    english_data_list, chinese_data_list, chinese_data_eval_list = data_list(config.data_path)
    english_data.process(english_data_list)
    for key, value in english_data.word_index.items():
        # print(key,value)
        f.writelines('{} {}'.format(key, value))
        f.writelines('\n')

if os.path.exists('data/{}.txt'.format('chinese_dict')):
    os.remove('data/{}.txt'.format('chinese_dict'))

with open('data/{}.txt'.format('chinese_dict'), 'a', encoding='utf-8') as f:
    english_data = Data('chinese')
    english_data_list, chinese_data_list, chinese_data_eval_list = data_list(config.data_path)
    english_data.process(chinese_data_list)
    for key, value in english_data.word_index.items():
        # print(key,value)
        f.writelines('{}\t{}'.format(key, value))
        f.writelines('\n')

# 定义模型
model = Transformers().to(config.device)

data = pro_data(data_path=config.data_path)
train_data, test_data, eval_data = Dataset(data, batch_size=5000, batch_num=10)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(config.epochs):
    for i, batch in enumerate(train_data):
        sum_loss = 0
        for line in batch:
            inputs_data, outputs_data, dec_output = line[0], line[1], line[2]
            inputs_data, outputs_data, dec_output = inputs_data.to(config.device), outputs_data.to(
                config.device), dec_output.to(config.device)
            dec_outputs = model(inputs_data.view(1, -1), outputs_data.view(1, -1))
            loss = criterion(dec_outputs, dec_output.view(-1))
            sum_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(sum_loss / 5))

    if epoch % 20 == 0:
        torch.save(model.state_dict(), 'model/model{}.pth'.format(epoch))
