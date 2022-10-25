#encoding=GBK
'''
合并主题
'''
from contribute_var import CONTRIBUTE , CONTRIBUTE_TOPICSIZE , CONTRIBUTE_DOC_DF , CONTRIBUTE_DOC_PERTOPIC
import pandas as pd
from get_topn_words import C_TFIDF_CALCULATION , EXTRACT_N_WORDS_PERTOPIC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#声明全局变量

data_df = pd.read_csv('../data/data_process.csv' , encoding = 'utf-8')
doc_df , topic_size , doc_per_topic = CONTRIBUTE(data_df)


doc_df['Doc_ID'] = range(len(doc_df))

c_tfidf , count = C_TFIDF_CALCULATION(doc_per_topic , num_data = len(doc_df))# tf矩阵

topic_similarity = cosine_similarity(c_tfidf.T)#主题相似度
np.fill_diagonal(topic_similarity, 0)  # 与自身的余弦相似度全部归0

def ADJUST(topic_to_merge , topic_to_merge_in) :
    '''
    更新全局变量
    :param topic_to_merge: 需要调整的主题编号
    :param topic_to_merge_in: 将该主题调整至的目标主题编号
    :return:
    '''
    #声明全局变量
    global data_df , doc_df , topic_size , doc_per_topic , c_tfidf , count , topic_similarity

    doc_df.loc[doc_df['Topic'] == topic_to_merge, "Topic"] = topic_to_merge_in  # 在文档集docs_df中改变该文档所对应的主题
    # 这一步只是将主题编号修改了，并未移除先前的主题序号

    # 更新docs_df
    old_topics = doc_df.sort_values("Topic").Topic.unique()
    map_topic = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}  # 将序号-1改设为现在的主题号
    doc_df.Topic = doc_df.Topic.map(map_topic)  # index : topic

    # 更新docs_per_topic
    doc_per_topic = doc_df.groupby(["Topic"], as_index=False).agg({"微博正文": ' '.join})

    # 更新c_tfidf矩阵,以及主题下对应的主题词
    c_tfidf, count = C_TFIDF_CALCULATION(doc_per_topic , num_data=len(doc_df))  # 更新tf矩阵

    # 更新topic_size
    topic_size = CONTRIBUTE_TOPICSIZE(doc_df)

    # 更新similarity矩阵
    topic_similarity = cosine_similarity(c_tfidf.T)  # 每个主题下单词分布两两之间的余弦相似度
    np.fill_diagonal(topic_similarity, 0)  # 与自身的余弦相似度全部归0

    print("主题", topic_to_merge, "被合并到主题:", topic_to_merge_in)
    print(topic_similarity.shape)

    topic_similarity


def MERGE_TOPIC(threshold = 0.5) :
    '''
    合并主题
    :param threshold: 主题相似度阈值，若两个主题相似度高于此值，则将其合并为一个主题
    :return: None:改变全局变量
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




