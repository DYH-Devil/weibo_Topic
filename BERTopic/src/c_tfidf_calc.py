#encoding=GBK
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents , num_data , ngram_range = (1 , 1)) :
    '''
    :param documents: �������Ӻ���ĵ�
    :param ngram_range: (1,1)
    :param num_data: �ĵ���Ŀ
    :return: c_tfidf , count
    '''
    stop_words = [i.strip() for i in open('../data/cn_stopwords.txt' , encoding = 'utf-8').readlines()]
    count = CountVectorizer(ngram_range = ngram_range ,
                            stop_words = stop_words)
    t = count.fit_transform(documents).toarray()# �����ĵ��Ĵ�Ƶ����
    # print("t shape: " , t.shape)

    # dict = count.vocabulary_ # �ʵ�
    sum = t.sum(axis = 1)# sumΪͳ��ÿһ���е��ʵ�����
    # print("sum shape: " , sum.shape)

    tf = np.divide(t.T , sum)# tfֵ:��Ƶ�ʣ���ʾÿ������ÿһ���еĳ���Ƶ��
    # print("tf shape: " , tf.shape)

    sum_t = t.sum(axis = 0)#�����������еĳ��ִ��������������ĵ��г��ֵĴ�����(ÿ���������ĵ����Ӷ���)
    # print("sum_t shape" , sum_t.shape)

    idf = np.log(1 + np.divide(num_data , sum_t)).reshape(-1 , 1)
    # print("idf shape: " , idf.shape)

    tf_idf = np.multiply(tf , idf)
    return tf_idf , count


def extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10) :
    '''
    :param c_tfidf: ���ʵ�c_tfidf����,���ϼ����shape[len(word_dic) , num_topic]
    :param count: �ʵ�ͳ��
    :param docs_per_topic: ���ĵ����������ӳɵĴ��ĵ�
    :param n_top: ������������ǰtopn�ĵ���
    :return:
    '''
    top_n_words = {}
    words = count.get_feature_names_out() # ���ʿ� list
    labels = list(docs_per_topic['Topic'])

    c_tfidf_transpose = c_tfidf.T# shape��[len(word_dic) , num_topic]ת��Ϊ[num_topic,len(word_dic)]
    # indices = c_tfidf_transpose.argsort()[:, -n_top:]
    # top_n_words = {label: [(words[j], c_tfidf_transpose[i][j]) for j in indices[i]][::-1] for i, label in
    #                enumerate(labels)}
    indexes = c_tfidf_transpose.argsort()[: , -n_top:]# ��c_tfidf��ֵ����ǰn_top�ߵ�Ԫ�أ������Ǵ�С����˳��
    for i , label in enumerate(labels) :# ÿ��������
        top_n_words[label] = []
        for index in indexes[i][::-1] :
            words_tfidf = [words[index] , c_tfidf_transpose[i][index]]
            top_n_words[label].append(words_tfidf)
    return top_n_words

def topic_size(docs_df) :
    '''
    :param docs_df: �ĵ���
    :return: ÿ�������µ��ĵ���Ŀ
    '''
    topic_size = (docs_df.groupby(['Topic']).
                  Doc.
                  count().
                  reset_index().
                  rename({"Topic" : "Topic" , "Doc" : "Size"} , axis = 'columns').
                  sort_values("Size" , ascending = False)
                  )
    return topic_size



