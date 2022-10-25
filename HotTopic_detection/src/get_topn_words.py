#encoding=GBK
'''
计算词的c_tfidf值
'''
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from contribute_var import CONTRIBUTE
import pandas as pd
import joblib

def C_TFIDF_CALCULATION(doc_per_topic , num_data , ngram_range = (1,1)) :
    '''
    计算c_tfidf值函数
    :param doc_per_topic: 按簇分组后连接的大文档
    :param num_doc: 文档数
    :param ngram_range: (1,1)
    :return: c_tfidf矩阵 , Count
    '''
    stop_words = [i.strip() for i in open('../data/cn_stopwords.txt', encoding='utf-8').readlines()]# 加载停用词
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words=stop_words)

    t = count.fit_transform(doc_per_topic['微博正文']).toarray()  # 计算文档的词频矩阵
    # print("t shape: " , t.shape)

    sum_c = t.sum(axis=1)  # sum为统计每一簇中单词的总数
    # print("sum_c shape: " , sum_c.shape)

    tf = np.divide(t.T, sum_c)  # tf值:词频率，表示每个词在每一簇中的出现频率
    # print("tf shape: " , tf.shape)

    sum_w = t.sum(axis=0)  # 词在所有类中的出现次数，即在所有文档中出现的次数，(每个类是由文档连接而得)
    # print("sum_w shape" , sum_w.shape)

    idf = np.log(1 + np.divide(num_data, sum_w)).reshape(-1, 1)
    # print("idf shape: " , idf.shape)

    tf_idf = np.multiply(tf, idf)


    return tf_idf , count


def EXTRACT_N_WORDS_PERTOPIC(c_tfidf , count , docs_per_topic , n_top = 10) :
    '''
    获取每个话题下的关键词
    :param c_tfidf: 通过计算得出的c_tfidf矩阵 shape[len(word_dict , topic_num)]
    :param count: 统计词典
    :param docs_per_topic: 按话题分组连接成的大文档
    :param n_top: 前n个热点词
    :return: topn_words
    '''
    top_n_words = {}

    word_spread = joblib.load('../data/word_spread_dict.dat')  # 加载词项热度词典
    spread_list = []
    for spread in word_spread.values():
        spread_list.append(spread)

    word_count = joblib.load('../data/word_count_dict.dat')  # 加载词频词频词典
    count_list = []
    for word in word_count.values():
        count_list.append(word)

    spread_array = np.array(spread_list).reshape(-1, 1)
    count_array = np.array(count_list).reshape(-1, 1)
    word_spread = np.divide(spread_array, count_array)  # 词项热度:spread / count

    tfidf_spread = np.multiply(c_tfidf, word_spread)  # c_tfidf值乘上词项热度权重

    words = count.get_feature_names_out() # 单词库 list
    tfidf_spread_T = tfidf_spread.T# 转置tfidf矩阵 shape[topicnum , len(word_dict)]

    labels = docs_per_topic['Topic'].to_list()
    indexes = tfidf_spread_T.argsort()[:, -n_top:]  # 在c_tfidf中值排名前n_top高的元素下标，但是是从小到大顺序
    for i, label in enumerate(labels):  # 每个话题下
        top_n_words[label] = []
        for index in indexes[i][::-1]:#从大到小
            words_tfidf = [words[index], tfidf_spread_T[i][index]]


            top_n_words[label].append(words_tfidf)

    return top_n_words


# if __name__ == '__main__':
#     hotopic = pd.read_csv('../data/data_process.csv', encoding='utf-8')
#     doc_df, topic_size, doc_per_topic = CONTRIBUTE(hotopic)  # 加载构建的三个变量
#     tfidf_spread , count = C_TFIDF_CALCULATION(doc_per_topic , len(doc_df))# c_tfidf矩阵
#
#     # print(tfidf_spread.shape) # [54591 , 56]
#     # print(tfidf_spread)
#     topn_words = EXTRACT_N_WORDS_PERTOPIC(c_tfidf = tfidf_spread ,
#                                           count = count ,
#                                           docs_per_topic = doc_per_topic ,
#                                           n_top = 10)
#
#     for topic , keywords in topn_words.items() :
#         print(topic , keywords)



