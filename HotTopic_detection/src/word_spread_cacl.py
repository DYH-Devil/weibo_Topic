#encoding=GBK
'''
用于计算每个词项的spread系数
'''
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib

def WORD_SPREAD_COUNT(hotopic) :
    '''
    :param hotopic:DataFrame
    :return: word_spread_dict:词项热度词典
    '''
    hotopic_text = hotopic['微博正文']
    hotopic_spread = hotopic['传播值'].astype(np.float)


    stop_words = [i.strip() for i in open('../data/cn_stopwords.txt' , encoding = 'utf-8').readlines()]
    count = CountVectorizer(ngram_range = (1 , 1) ,
                            stop_words = stop_words)

    count.fit_transform(hotopic_text)
    VOCAB = count.vocabulary_.keys()
    print("len vocab:" , len(VOCAB))

    Word_Count = {}
    for key in VOCAB :
        Word_Count[key] = 0#初始化Count词典

    Spread_Dict = {}
    for key in VOCAB :
        Spread_Dict[key] = 0 #初始化Spread词典
    print("Spread_Dict len:" , len(Spread_Dict))
    # print(Spread_Dict)

    #统计词频和词热度
    for index , line in enumerate(hotopic_text) :
        word_list = line.split(' ')
        for word in word_list :
            Spread_Dict[word] += hotopic_spread.iloc[index]
            Word_Count[word] += 1
    return Spread_Dict , Word_Count

if __name__ == '__main__':
    # 读取文件
    hotopic = pd.read_csv('../data/data_process.csv', encoding='utf-8')
    word_spread_dict  , word_count_dict = WORD_SPREAD_COUNT(hotopic)

    print(len(word_spread_dict))
    print(len(word_count_dict))


    joblib.dump(word_spread_dict , '../data/word_spread_dict.dat')
    joblib.dump(word_count_dict , '../data/word_count_dict.dat')



