#encoding=GBK
'''
�ϲ�����
'''
from contribute_var import CONTRIBUTE , CONTRIBUTE_TOPICSIZE , CONTRIBUTE_DOC_DF , CONTRIBUTE_DOC_PERTOPIC
import pandas as pd
from get_topn_words import C_TFIDF_CALCULATION , EXTRACT_N_WORDS_PERTOPIC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#����ȫ�ֱ���

data_df = pd.read_csv('../data/data_process.csv' , encoding = 'utf-8')
doc_df , topic_size , doc_per_topic = CONTRIBUTE(data_df)


doc_df['Doc_ID'] = range(len(doc_df))

c_tfidf , count = C_TFIDF_CALCULATION(doc_per_topic , num_data = len(doc_df))# tf����

topic_similarity = cosine_similarity(c_tfidf.T)#�������ƶ�
np.fill_diagonal(topic_similarity, 0)  # ��������������ƶ�ȫ����0

def ADJUST(topic_to_merge , topic_to_merge_in) :
    '''
    ����ȫ�ֱ���
    :param topic_to_merge: ��Ҫ������������
    :param topic_to_merge_in: ���������������Ŀ��������
    :return:
    '''
    #����ȫ�ֱ���
    global data_df , doc_df , topic_size , doc_per_topic , c_tfidf , count , topic_similarity

    doc_df.loc[doc_df['Topic'] == topic_to_merge, "Topic"] = topic_to_merge_in  # ���ĵ���docs_df�иı���ĵ�����Ӧ������
    # ��һ��ֻ�ǽ��������޸��ˣ���δ�Ƴ���ǰ���������

    # ����docs_df
    old_topics = doc_df.sort_values("Topic").Topic.unique()
    map_topic = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}  # �����-1����Ϊ���ڵ������
    doc_df.Topic = doc_df.Topic.map(map_topic)  # index : topic

    # ����docs_per_topic
    doc_per_topic = doc_df.groupby(["Topic"], as_index=False).agg({"΢������": ' '.join})

    # ����c_tfidf����,�Լ������¶�Ӧ�������
    c_tfidf, count = C_TFIDF_CALCULATION(doc_per_topic , num_data=len(doc_df))  # ����tf����

    # ����topic_size
    topic_size = CONTRIBUTE_TOPICSIZE(doc_df)

    # ����similarity����
    topic_similarity = cosine_similarity(c_tfidf.T)  # ÿ�������µ��ʷֲ�����֮����������ƶ�
    np.fill_diagonal(topic_similarity, 0)  # ��������������ƶ�ȫ����0

    print("����", topic_to_merge, "���ϲ�������:", topic_to_merge_in)
    print(topic_similarity.shape)

    topic_similarity


def MERGE_TOPIC(threshold = 0.5) :
    '''
    �ϲ�����
    :param threshold: �������ƶ���ֵ���������������ƶȸ��ڴ�ֵ������ϲ�Ϊһ������
    :return: None:�ı�ȫ�ֱ���
    '''
    i = 0
    while (i < topic_similarity.shape[0]) :
        for row_index , row in enumerate(topic_similarity) :
            max_sim = max(row)
            if(max_sim >= threshold) :
                t1 = row_index - 1
                t2 = list(row).index(max_sim) - 1
                ADJUST(t1 , t2)
                break
            else :
                continue
        i += 1
    topic_words = EXTRACT_N_WORDS_PERTOPIC(c_tfidf , count , doc_per_topic , n_top = 10)
    return doc_df , topic_size , doc_per_topic , c_tfidf , topic_similarity , topic_words




