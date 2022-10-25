#encoding=GBK
'''
����ʵ�c_tfidfֵ
'''
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from contribute_var import CONTRIBUTE
import pandas as pd
import joblib

def C_TFIDF_CALCULATION(doc_per_topic , num_data , ngram_range = (1,1)) :
    '''
    ����c_tfidfֵ����
    :param doc_per_topic: ���ط�������ӵĴ��ĵ�
    :param num_doc: �ĵ���
    :param ngram_range: (1,1)
    :return: c_tfidf���� , Count
    '''
    stop_words = [i.strip() for i in open('../data/cn_stopwords.txt', encoding='utf-8').readlines()]# ����ͣ�ô�
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words=stop_words)

    t = count.fit_transform(doc_per_topic['΢������']).toarray()  # �����ĵ��Ĵ�Ƶ����
    # print("t shape: " , t.shape)

    sum_c = t.sum(axis=1)  # sumΪͳ��ÿһ���е��ʵ�����
    # print("sum_c shape: " , sum_c.shape)

    tf = np.divide(t.T, sum_c)  # tfֵ:��Ƶ�ʣ���ʾÿ������ÿһ���еĳ���Ƶ��
    # print("tf shape: " , tf.shape)

    sum_w = t.sum(axis=0)  # �����������еĳ��ִ��������������ĵ��г��ֵĴ�����(ÿ���������ĵ����Ӷ���)
    # print("sum_w shape" , sum_w.shape)

    idf = np.log(1 + np.divide(num_data, sum_w)).reshape(-1, 1)
    # print("idf shape: " , idf.shape)

    tf_idf = np.multiply(tf, idf)


    return tf_idf , count


def EXTRACT_N_WORDS_PERTOPIC(c_tfidf , count , docs_per_topic , n_top = 10) :
    '''
    ��ȡÿ�������µĹؼ���
    :param c_tfidf: ͨ������ó���c_tfidf���� shape[len(word_dict , topic_num)]
    :param count: ͳ�ƴʵ�
    :param docs_per_topic: ������������ӳɵĴ��ĵ�
    :param n_top: ǰn���ȵ��
    :return: topn_words
    '''
    top_n_words = {}

    word_spread = joblib.load('../data/word_spread_dict.dat')  # ���ش����ȶȴʵ�
    spread_list = []
    for spread in word_spread.values():
        spread_list.append(spread)

    word_count = joblib.load('../data/word_count_dict.dat')  # ���ش�Ƶ��Ƶ�ʵ�
    count_list = []
    for word in word_count.values():
        count_list.append(word)

    spread_array = np.array(spread_list).reshape(-1, 1)
    count_array = np.array(count_list).reshape(-1, 1)
    word_spread = np.divide(spread_array, count_array)  # �����ȶ�:spread / count

    tfidf_spread = np.multiply(c_tfidf, word_spread)  # c_tfidfֵ���ϴ����ȶ�Ȩ��

    words = count.get_feature_names_out() # ���ʿ� list
    tfidf_spread_T = tfidf_spread.T# ת��tfidf���� shape[topicnum , len(word_dict)]

    labels = docs_per_topic['Topic'].to_list()
    indexes = tfidf_spread_T.argsort()[:, -n_top:]  # ��c_tfidf��ֵ����ǰn_top�ߵ�Ԫ���±꣬�����Ǵ�С����˳��
    for i, label in enumerate(labels):  # ÿ��������
        top_n_words[label] = []
        for index in indexes[i][::-1]:#�Ӵ�С
            words_tfidf = [words[index], tfidf_spread_T[i][index]]


            top_n_words[label].append(words_tfidf)

    return top_n_words


# if __name__ == '__main__':
#     hotopic = pd.read_csv('../data/data_process.csv', encoding='utf-8')
#     doc_df, topic_size, doc_per_topic = CONTRIBUTE(hotopic)  # ���ع�������������
#     tfidf_spread , count = C_TFIDF_CALCULATION(doc_per_topic , len(doc_df))# c_tfidf����
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



