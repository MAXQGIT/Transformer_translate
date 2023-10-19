from transformer import Transformers
import torch
#需要翻译的文本
input_data = 'i love you'

#数据中各个字符对应的编码
english_data = {line.split()[0]: line.split()[1] for line in
                open('data/english_dict.txt', 'r', encoding='utf-8').readlines()}
chinese_data = {line.split('\t')[0]: int(line.split('\t')[1]) for line in
                open('data/chinese_dict.txt', 'r', encoding='utf-8').readlines()}
index_word = {j:i for i,j in chinese_data.items()}

#将输入的数据转换为对应的编码
input_data_list = []
for word in input_data.split():
    if word in english_data.keys():
        input_data_list.append(int(english_data[word]))

input_data_tensor = torch.LongTensor(input_data_list).view(1, -1)


#加载和导入模型
model = Transformers()
model.load_state_dict(torch.load('model/model80.pth'))
next_symbol = chinese_data['S']

#定义文本翻译的预测函数
def greedy_decoder(model, input_data_tensor, start_symbol):
    enc_outputs = model.encoder(input_data_tensor)
    dec_input = torch.zeros(1, 0).type_as(input_data_tensor.data)
    next_symbol = start_symbol

    terminal = False
    while not terminal:
        dec_input = torch.cat([dec_input, torch.tensor([[next_symbol]], dtype=input_data_tensor.dtype)], -1)
        dec_outpus = model.decoder(input_data_tensor, enc_outputs, dec_input)
        projected = model.propetice(dec_outpus)

        prob = projected.squeeze(0).max(dim=-1)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == chinese_data['E']:
            terminal = True

    predict_data = dec_input[:, 1:]
    return predict_data


enc_decoder = greedy_decoder(model, input_data_tensor, next_symbol)
for i in enc_decoder.squeeze():
    print(index_word[i.item()])

print(input_data+'->'+''.join([index_word[i.item()] for i in enc_decoder.squeeze()]))