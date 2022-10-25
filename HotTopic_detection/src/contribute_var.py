#encoding=GBK
'''
构建一些变量供后续使用
'''

import pandas as pd
import joblib

def CONTRIBUTE_DOC_DF(data_df) :
    '''
    :param data_df: 微博文档集
    :return: 每个微博以及各自所属的簇
    '''
    cluster_res = joblib.load('../model/sentence_cluster.dat')#加载聚类结果

    data_df['text_ID'] = range(len(data_df))
    data_df['Topic'] = cluster_res.labels_
    return data_df

def CONTRIBUTE_TOPICSIZE(data_df) :
    '''
    :param data_df: 带有topic的微博文档集
    :return: 每个话题下的文档数
    '''
    topic_size = (data_df.groupby(['Topic']).
                  微博正文.
                  count().
                  reset_index().
                  rename({"Topic": "Topic", "微博正文": "Size"}, axis='columns').
                  sort_values("Size", ascending=True)
                  )
    return topic_size

def CONTRIBUTE_DOC_PERTOPIC(data_df) :
    '''
    :param data_df: 微博文档集
    :return: 按话题将文档连接成为一个大文档
    '''
    doc_per_topic = data_df.groupby(['Topic'], as_index=False).agg({"微博正文": ' '.join})
    return doc_per_topic


def CONTRIBUTE(data_df) :
    '''
    :param data_df:微博文档集
    :return: 四个变量
    '''
    doc_df = CONTRIBUTE_DOC_DF(data_df)

    topic_size = CONTRIBUTE_TOPICSIZE(doc_df)

    doc_per_topic = CONTRIBUTE_DOC_PERTOPIC(doc_df)

    return doc_df , topic_size , doc_per_topic



