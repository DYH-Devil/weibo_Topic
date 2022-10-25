#encoding=GBK
'''
����һЩ����������ʹ��
'''

import pandas as pd
import joblib

def CONTRIBUTE_DOC_DF(data_df) :
    '''
    :param data_df: ΢���ĵ���
    :return: ÿ��΢���Լ����������Ĵ�
    '''
    cluster_res = joblib.load('../model/sentence_cluster.dat')#���ؾ�����

    data_df['text_ID'] = range(len(data_df))
    data_df['Topic'] = cluster_res.labels_
    return data_df

def CONTRIBUTE_TOPICSIZE(data_df) :
    '''
    :param data_df: ����topic��΢���ĵ���
    :return: ÿ�������µ��ĵ���
    '''
    topic_size = (data_df.groupby(['Topic']).
                  ΢������.
                  count().
                  reset_index().
                  rename({"Topic": "Topic", "΢������": "Size"}, axis='columns').
                  sort_values("Size", ascending=True)
                  )
    return topic_size

def CONTRIBUTE_DOC_PERTOPIC(data_df) :
    '''
    :param data_df: ΢���ĵ���
    :return: �����⽫�ĵ����ӳ�Ϊһ�����ĵ�
    '''
    doc_per_topic = data_df.groupby(['Topic'], as_index=False).agg({"΢������": ' '.join})
    return doc_per_topic


def CONTRIBUTE(data_df) :
    '''
    :param data_df:΢���ĵ���
    :return: �ĸ�����
    '''
    doc_df = CONTRIBUTE_DOC_DF(data_df)

    topic_size = CONTRIBUTE_TOPICSIZE(doc_df)

    doc_per_topic = CONTRIBUTE_DOC_PERTOPIC(doc_df)

    return doc_df , topic_size , doc_per_topic



