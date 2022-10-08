#encoding=GBK
import numpy as np

from data_process import data_process
import pandas as pd
from cluster import ump_sentence_res
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents , num_data , ngram_range = (1 , 1)) :
    '''
    :param documents: 按簇连接后的文档
    :param ngram_range: (1,1)
    :param num_data: 文档数目
    :return: c_tfidf , count
    '''
    stop_words = [i.strip() for i in open('../data/cn_stopwords.txt' , encoding = 'utf-8').readlines()]
    count = CountVectorizer(ngram_range = ngram_range ,
                            stop_words = stop_words)
    t = count.fit_transform(documents).toarray()# 计算文档的词频矩阵
    print("t shape: " , t.shape)

    # dict = count.vocabulary_ # 词典
    sum = t.sum(axis = 1)# sum为统计每一簇中单词的总数
    print("sum shape: " , sum.shape)

    tf = np.divide(t.T , sum)# tf值:词频率，表示每个词在每一簇中的出现频率
    print("tf shape: " , tf.shape)

    sum_t = t.sum(axis = 0)#词在所有类中的出现次数，即在所有文档中出现的次数，(每个类是由文档连接而得)
    print("sum_t shape" , sum_t.shape)

    idf = np.log(1 + np.divide(num_data , sum_t)).reshape(-1 , 1)
    print("idf shape: " , idf.shape)

    tf_idf = np.multiply(tf , idf)
    return tf_idf , count


def extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10) :
    '''
    :param c_tfidf: 单词的c_tfidf矩阵,由上计算得shape[len(word_dic) , num_topic]
    :param count: 词典统计
    :param docs_per_topic: 由文档按主题连接成的大文档
    :param n_top: 参数，主题下前topn的单词
    :return:
    '''
    top_n_words = {}
    words = count.get_feature_names_out() # 单词库
    labels = list(docs_per_topic['Topic'])

    c_tfidf_transpose = c_tfidf.T# shape由[len(word_dic) , num_topic]转置为[num_topic,len(word_dic)]
    # indices = c_tfidf_transpose.argsort()[:, -n_top:]
    # top_n_words = {label: [(words[j], c_tfidf_transpose[i][j]) for j in indices[i]][::-1] for i, label in
    #                enumerate(labels)}
    indexes = c_tfidf_transpose.argsort()[: , -n_top:]# 在c_tfidf中值排名前n_top高的元素，但是是从小到达顺序
    for i , label in enumerate(labels) :# 每个话题下
        top_n_words[label] = []
        for index in indexes[i][::-1] :
            words_tfidf = [words[index] , c_tfidf_transpose[i][index]]
            top_n_words[label].append(words_tfidf)
    return top_n_words

docs = data_process()

docs_df = pd.DataFrame(docs ,columns = ['Doc'])
docs_df['Topic'] = ump_sentence_res['label']
docs_df['Doc_ID'] = range(len(docs_df))
print(docs_df)

docs_per_topic = docs_df.groupby(['Topic'] , as_index = False).agg({"Doc" : ' '.join})#将所有文档按主题分别连接为一个大文档
print(docs_per_topic)

print("number of topic: " , len(docs_per_topic['Doc'].tolist()))#话题数


c_tfidf , count = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# tf矩阵
print(c_tfidf)
print("c_tfidf shape: " , c_tfidf.shape)

topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)

del topic_words[-1]
for topic , words in topic_words.items() :
    print(topic , words)

