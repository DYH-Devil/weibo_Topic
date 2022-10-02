import jieba 
import csv
import re
from string import punctuation as punctuation_string
from copy import deepcopy

def text_process(text_list) :
    text_clean = []
    for text in text_list :
        #-------先把英文标点转为中文标点------------
        text = text.replace("?", "？")
        text = text.replace(".", "。")
        text = text.replace(",", "，")
        #---------------------------------------

        filters = ['\t', '\n', '\x97', '\x96' , '$', '%', '&', '@' , '！', '，', '？', '——', '_', '。', '—', '…', '；', '“',
                   "”", '：','（','）','《','》', '、', '·']
        text = re.sub('|'.join(filters), ' ', text)  # |表示综合所有的filters,全部替换
        text = re.sub("【.+?】", "", text)  # 去除【】里的内容，通常里面的内容是引用
        text = re.sub('http?://\S+|www\.\S+', '', text)  # 去除url链接
        text = re.sub("#.*?#", "", text)  # 去除话题引用
        text = re.sub("<.*?>", "", text)#去除html有关字符
        text = re.sub("\n", "", text)
        text = re.sub('[0-9]', '', text)#去除数字
        text = re.sub('[a-zA-Z]' , '' , text)#去除英文字符
        text = text.replace('*', '')
        if(len(text) > 1) :
            text_clean.append(text)

    return text_clean



def word_split(text_clean) :
    '''
    :param text_clean: 清洗后的文本
    :return:
    '''
    text_split_list = []#分词后的总表
    for text in text_clean:
        text_split = jieba.lcut(text)#对每个句子进行分词
        text_split_copy = deepcopy(text_split)#remove前需要深拷贝，否则踩大坑
        for word in text_split_copy :
            if(word == ' ' or len(word) < 2) :#删除空格和长度小于2的词语
                text_split.remove(word)

        if (len(text_split) > 2) :
            text_split_list.append(text_split)#text_split_list:[['x','x'...],['x','x'...]...]
    return text_split_list



def del_stopwords(text_split_list) :
    '''
    :param text_split_list: 分词后的文本
    :return:
    '''
    word_dict = word_count(text_split_list)#分完词后的词频统计词典

    stop_words = open('../../data/cn_stopwords.txt', 'r', encoding='utf-8').readlines()  # 读取停用词库
    stop_list = []  # 将文件中的停用词存入列表
    for i in stop_words:
        stop_list.append(i.strip())

    for text in text_split_list :
        text_copy = deepcopy(text)
        for word in text_copy :
            if(word in stop_list or word_dict[word] < 5 ) :
                text.remove(word)
    text_split_list_copy = deepcopy(text_split_list)
    for text in text_split_list_copy :
        if(len(text) < 3 ) :
            text_split_list.remove(text)
    return text_split_list


def word_count(text_list) :#计算词语文档频率
    word_list = []
    for text in text_list :
        for word in text :
            if(word not in word_list) :
                word_list.append(word)#所有词语集合

    word_dict = {}
    for word in word_list :
        word_dict[word] = 0
    for word in word_list :
        for text in text_list :
            if(word in text) :
                word_dict[word] += 1
    return word_dict

file = open('../../data/content.csv' , 'r' , encoding = 'utf-8')
reader = csv.reader(file)

res_file = open('../sample-data/res_file.dat' , 'w' , encoding = 'utf-8')

text_list = []
for line in reader :
    text_list.append(line[1])

text_list.pop(0)

text_clean = text_process(text_list)
# print(text_clean)
text_split_list = word_split(text_clean)
#print(text_split_list)
text_del_stopwords = del_stopwords(text_split_list)



for text in text_del_stopwords :
    res_file.write(' '.join(text)+'\n')





