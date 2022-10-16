#encoding=GBK

'''
����Ԥ����
������ϴ���ִʣ�ȥͣ��
'''

import re
import jieba

#������ϴ
def DATA_CLEAN(text_list) :
    text_clean = []
    for text in text_list :
        #-------�Ȱ�Ӣ�ı��תΪ���ı��------------
        text = text.replace("?", "��")
        text = text.replace(".", "��")
        text = text.replace(",", "��")
        text = text.replace("[" , "{")
        text = text.replace("]" , "}")
        #---------------------------------------

        filters = ['\t', '\n', '\x97', '\x96' , '$', '%', '&', '@' , '��', '��', '��', '����', '_', '��', '��', '��', '��', '��','-',
                   "��", '��','��','��','��','��', '��', '��','~','(',')','!',':','/','��','��']
        text = re.sub("{.+?}","",text)
        text = re.sub('|'.join(filters), '', text)  # |��ʾ�ۺ����е�filters,ȫ���滻
        text = re.sub("��.+?��", "", text)  # ȥ������������ݣ�ͨ�����������������
        text = re.sub('http?://\S+|www\.\S+', '', text)  # ȥ��url����
        text = re.sub("#.*?#", "", text)  # ȥ����������
        text = re.sub("<.*?>", "", text)#ȥ��html�й��ַ�
        text = re.sub("\n", "", text)
        text = re.sub('[0-9]', '', text)#ȥ������
        text = re.sub('[a-zA-Z]' , '' , text)#ȥ��Ӣ���ַ�
        text = text.replace('*', '')
        text = text.replace(" " , "")
        if(len(text) > 4) :
            text_clean.append(text)

    return text_clean

#�ִ�
def TEXT_SPLIT(text_list) :
    text_split = []
    for text in text_list :
        text = jieba.lcut(text)
        text_split.append(text)
    return text_split

#ȥͣ�ô�
def DEL_STOPWORDS(text_list) :
    stop_file = open('../data/cn_stopwords.txt' , 'r' , encoding = 'utf-8')
    stop_words = []
    for word in stop_file.readlines():#����ͣ�ô�
        stop_words.append(word.strip())

    text_del_stop = []
    for text in text_list:
        text_clean_stop = []
        for word in text:
            if word not in stop_words and len(word) > 1:
                text_clean_stop.append(word)
        if (len(text_clean_stop) > 3):
            text_del_stop.append(' '.join(text_clean_stop))
    return text_del_stop


def DATA_PROCESS(data_df) :
    '''
    :param data_Df: ΢���ĵ��� DataFrame
    :return: Ԥ�������΢�����ݼ�
    '''
    data_df['text'] = data_df['text'].astype(str)
    text = data_df['text'].to_list()
    text_clean = DATA_CLEAN(text)
    text_split = TEXT_SPLIT(text_clean)
    text_del_stopwords = DEL_STOPWORDS(text_split)
    return text_del_stopwords

