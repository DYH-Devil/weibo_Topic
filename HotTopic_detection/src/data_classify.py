#encoding=GBK
'''
���ļ���ʱ������
'''

import pandas as pd
import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def change(date) :
    date_change = datetime.datetime.strptime(date , "%Y/%m/%d %H:%M")
    date_change = str(date_change)
    return date_change


#��΢�����з���(�ȵ㻰�� �� ���ȵ㻰��)
def WEIBO_CLASSIFY(data_df , alpha = 0.7 , beta = 0.2 , theta = 0.1) :
    '''
    :param data_df: ΢���ĵ���:DataFrame
    :param alpha: ΢��ת����Ȩ��
    :param beta: ΢��������Ȩ��
    :param theta: ΢��������Ȩ��
    :return: None �������΢�����ݼ�ֱ�ӱ���
    '''
    data_df = data_df[['bid', 'user_id', '�û��ǳ�', '΢������', 'ת����', '������', '������', '����ʱ��']]
    #ȥ��
    data_df.drop_duplicates(inplace = True)

    #�ȶ�΢�����ݼ������ڽ�������
    data_df['����ʱ��'] = data_df['����ʱ��'].map(change)
    data_df = data_df.loc[(data_df['����ʱ��'] >= '2022-10-17 00:00:00') & (data_df['����ʱ��'] <= '2022-10-19 23:59:59')]
    data_df = data_df.sort_values(by='����ʱ��')

    # ����������Ϊ�յ�ֵ���0
    data_df['ת����'].fillna(0 , inplace = True)
    data_df['������'].fillna(0 , inplace = True)
    data_df['������'].fillna(0 , inplace = True)

    #΢��spread�ȶ�ֵ���㹫ʽ
    data_df['����ֵ'] = alpha * np.log(data_df['ת����']) + \
                     beta * np.log(data_df['������']) + \
                     theta * np.log(data_df['������'])

    data_df.loc[data_df['����ֵ'] == -np.inf] = 0

    hot_topic = data_df.loc[(data_df['����ֵ'] > 0)]



    hot_topic.to_csv('../data/hotopic.csv' , index = False)#�������Ż�������

