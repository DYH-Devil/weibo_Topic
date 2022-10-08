'''
data_process:对数据进行简单处理
'''

from clean_function import text_process
import pandas as pd
import jieba
import joblib

def data_process() :
    '''
    简单处理文本
    :return: 分词，清洗后的语料
    '''
    data_df = pd.read_csv('../data/content.csv' , encoding = 'GBK')
    stop_file = open('../data/cn_stopwords.txt' , 'r' , encoding = 'utf-8')

    stop_words = []
    for word in stop_file.readlines() :
        stop_words.append(word.strip())

    text_list = []
    for line in data_df["微博正文"] :
        text_list.append(line)

    print(len(text_list))

    text_clean = text_process(text_list)#对文本进行简单清洗

    text_split = []
    for text in text_clean :
        text = jieba.lcut(text)
        text_split.append(text)

    text_stop = []
    for text in text_split :
        text_clean_stop = []
        for word in text :
            if word not in stop_words and len(word) > 1 :
                text_clean_stop.append(word)
        if(len(text_clean_stop) > 3) :
            text_stop.append(' '.join(text_clean_stop))

    return text_stop

