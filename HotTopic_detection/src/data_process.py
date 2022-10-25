#encoding=GBK
'''
����Ԥ����
'''

import os
import pandas as pd
import re
import warnings
import jieba

warnings.filterwarnings('ignore')

def DATA_CLEAN(data_df) :
    data_df['΢������'] = data_df['΢������'].astype(str)
    text_clean = []
    text_spread = []
    text_repostsnum = []
    text_commentsnum = []
    text_attitudesnum = []
    for index , text in enumerate(data_df['΢������'].to_list()):
        # -------�Ȱ�Ӣ�ı��תΪ���ı��------------
        text = text.replace("?", "��")
        text = text.replace(".", "��")
        text = text.replace(",", "��")
        text = text.replace("[", "{")
        text = text.replace("]", "}")
        text = text.replace("+", "#")
        # ---------------------------------------

        filters = ['\t', '\n', '\x97', '\x96', '$', '%', '&', '@', '��', '��', '��', '����', '_', '��', '��', '��', '��', '��',
                   '-', "��", '��', '��', '��', '��', '��', '��', '��', '~', '(', ')', '!', ':', '/', '��', '��']
        text = re.sub("{.+?}", "", text)
        text = re.sub('|'.join(filters), '', text)  # |��ʾ�ۺ����е�filters,ȫ���滻
        text = re.sub("��.+?��", "", text)  # ȥ������������ݣ�ͨ�����������������
        text = re.sub('http?://\S+|www\.\S+', '', text)  # ȥ��url����
        text = re.sub("#.*?#", "", text)  # ȥ����������
        text = re.sub("<.*?>", "", text)  # ȥ��html�й��ַ�
        text = re.sub("\n", "", text)
        text = re.sub('[0-9]', '', text)  # ȥ������
        text = re.sub('[a-zA-Z]', '', text)  # ȥ��Ӣ���ַ�
        text = text.replace('*', '')
        text = text.replace(" ", "")
        if (len(text) > 3):
            text_clean.append(text)
            text_repostsnum.append(data_df['ת����'].iloc[index])
            text_commentsnum.append(data_df['������'].iloc[index])
            text_attitudesnum.append(data_df['������'].iloc[index])
            text_spread.append(data_df['����ֵ'].iloc[index])

    text_list = list(zip(text_clean, text_repostsnum , text_commentsnum , text_attitudesnum , text_spread))
    data_clean_pd = pd.DataFrame(text_list , columns=['΢������' , 'ת����' , '������' , '������' , '����ֵ'])
    return data_clean_pd


#�ִ�
def TEXT_SPLIT(data_df) :
    jieba.load_userdict('../data/jieba_userdict.txt')
    text_split = []
    for text in data_df['΢������'].to_list() :
        text = jieba.lcut(text)
        text_split.append(text)
    text_split_list = list(zip(text_split , data_df['ת����'] , data_df['������'] , data_df['������'] , data_df['����ֵ']))
    text_split_df = pd.DataFrame(text_split_list , columns = ['΢������' , 'ת����' , '������' , '������' , '����ֵ'])
    return text_split_df


#ȥͣ�ô�
def DEL_STOPWORDS(data_df) :
    stop_file = open('../data/cn_stopwords.txt' , 'r' , encoding = 'utf-8')
    stop_words = []
    for word in stop_file.readlines():#����ͣ�ô�
        stop_words.append(word.strip())

    text_del_stop = []
    text_repostsnum = []
    text_commentsnum = []
    text_attitudesnum = []
    text_spread = []

    for index , text in enumerate(data_df['΢������'].to_list()):
        text_clean_stop = []
        for word in text:
            if word not in stop_words and len(word) > 1:
                text_clean_stop.append(word)
        if (len(text_clean_stop) > 3):
            text_del_stop.append(' '.join(text_clean_stop))
            text_repostsnum.append(data_df['ת����'].iloc[index])
            text_commentsnum.append(data_df['������'].iloc[index])
            text_attitudesnum.append(data_df['������'].iloc[index])
            text_spread.append(data_df['����ֵ'].iloc[index])
    text_stopdel_list = list(zip(text_del_stop , text_repostsnum , text_commentsnum , text_attitudesnum , text_spread))
    text_stopdel_pd = pd.DataFrame(text_stopdel_list , columns = ['΢������' , 'ת����' , '������' , '������' , '����ֵ'])
    return text_stopdel_pd


def DATA_PROCESS(data_df) :
    '''
    :param data_df: ΢���ĵ�:dataframe
    :return:
    '''
    data_clean_pd = DATA_CLEAN(data_df)
    text_split_df = TEXT_SPLIT(data_clean_pd)
    text_stopdel_pd = DEL_STOPWORDS(text_split_df)
    return text_stopdel_pd