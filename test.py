
import re
str = 'doi:10.1016/0012825266900298'

a =re.sub('(.$)',r'\1-',str)
print(a)
#output
'doi:10.1016/001282526690029-8'

b = re.sub('(10)',r'\1***',str,count=1)
print(b)
#output
'doi***:10.1016/0012825266900298'
a ='  194'
b = a.split(' ')
print(len(b))
for i in b:
    print(i)


from utils import normalizeString, cht_to_chs

def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        english_data, chinese_data, chinese_data_eval = [], [], []
        for line in f.readlines():
            english_data.append(normalizeString(line.split('\t')[0]))
            chinese_data.append('S' + cht_to_chs(line.split('\t')[1]))
            chinese_data_eval.append(cht_to_chs(line.split('\t')[1] + 'E'))

        return english_data, chinese_data, chinese_data_eval

data_path = 'data/en-cn.txt'

a=  ['我 DJ','斯卡萨的的','是的的 萨克回答说的 阿斯顿']
b = [line.replace(' ','') for line in a]
print(b)