#encoding=GBK
'''
数据预处理
'''

import os
import pandas as pd
import re
import warnings
import jieba

warnings.filterwarnings('ignore')

def DATA_CLEAN(data_df) :
    data_df['微博正文'] = data_df['微博正文'].astype(str)
    text_clean = []
    text_spread = []
    text_repostsnum = []
    text_commentsnum = []
    text_attitudesnum = []
    for index , text in enumerate(data_df['微博正文'].to_list()):
        # -------先把英文标点转为中文标点------------
        text = text.replace("?", "？")
        text = text.replace(".", "。")
        text = text.replace(",", "，")
        text = text.replace("[", "{")
        text = text.replace("]", "}")
        text = text.replace("+", "#")
        # ---------------------------------------

        filters = ['\t', '\n', '\x97', '\x96', '$', '%', '&', '@', '！', '，', '？', '——', '_', '。', '—', '…', '；', '“',
                   '-', "”", '：', '（', '）', '《', '》', '、', '·', '~', '(', ')', '!', ':', '/', '＃', '─']
        text = re.sub("{.+?}", "", text)
        text = re.sub('|'.join(filters), '', text)  # |表示综合所有的filters,全部替换
        text = re.sub("【.+?】", "", text)  # 去除【】里的内容，通常里面的内容是引用
        text = re.sub('http?://\S+|www\.\S+', '', text)  # 去除url链接
        text = re.sub("#.*?#", "", text)  # 去除话题引用
        text = re.sub("<.*?>", "", text)  # 去除html有关字符
        text = re.sub("\n", "", text)
        text = re.sub('[0-9]', '', text)  # 去除数字
        text = re.sub('[a-zA-Z]', '', text)  # 去除英文字符
        text = text.replace('*', '')
        text = text.replace(" ", "")
        if (len(text) > 3):
            text_clean.append(text)
            text_repostsnum.append(data_df['转发数'].iloc[index])
            text_commentsnum.append(data_df['评论数'].iloc[index])
            text_attitudesnum.append(data_df['点赞数'].iloc[index])
            text_spread.append(data_df['传播值'].iloc[index])

    text_list = list(zip(text_clean, text_repostsnum , text_commentsnum , text_attitudesnum , text_spread))
    data_clean_pd = pd.DataFrame(text_list , columns=['微博正文' , '转发数' , '评论数' , '点赞数' , '传播值'])
    return data_clean_pd


#分词
def TEXT_SPLIT(data_df) :
    jieba.load_userdict('../data/jieba_userdict.txt')
    text_split = []
    for text in data_df['微博正文'].to_list() :
        text = jieba.lcut(text)
        text_split.append(text)
    text_split_list = list(zip(text_split , data_df['转发数'] , data_df['评论数'] , data_df['点赞数'] , data_df['传播值']))
    text_split_df = pd.DataFrame(text_split_list , columns = ['微博正文' , '转发数' , '评论数' , '点赞数' , '传播值'])
    return text_split_df


#去停用词
def DEL_STOPWORDS(data_df) :
    stop_file = open('../data/cn_stopwords.txt' , 'r' , encoding = 'utf-8')
    stop_words = []
    for word in stop_file.readlines():#加载停用词
        stop_words.append(word.strip())

    text_del_stop = []
    text_repostsnum = []
    text_commentsnum = []
    text_attitudesnum = []
    text_spread = []

    for index , text in enumerate(data_df['微博正文'].to_list()):
        text_clean_stop = []
        for word in text:
            if word not in stop_words and len(word) > 1:
                text_clean_stop.append(word)
        if (len(text_clean_stop) > 3):
            text_del_stop.append(' '.join(text_clean_stop))
            text_repostsnum.append(data_df['转发数'].iloc[index])
            text_commentsnum.append(data_df['评论数'].iloc[index])
            text_attitudesnum.append(data_df['点赞数'].iloc[index])
            text_spread.append(data_df['传播值'].iloc[index])
    text_stopdel_list = list(zip(text_del_stop , text_repostsnum , text_commentsnum , text_attitudesnum , text_spread))
    text_stopdel_pd = pd.DataFrame(text_stopdel_list , columns = ['微博正文' , '转发数' , '评论数' , '点赞数' , '传播值'])
    return text_stopdel_pd


def DATA_PROCESS(data_df) :
    '''
    :param data_df: 微博文档:dataframe
    :return:
    '''
    data_clean_pd = DATA_CLEAN(data_df)
    text_split_df = TEXT_SPLIT(data_clean_pd)
    text_stopdel_pd = DEL_STOPWORDS(text_split_df)
    return text_stopdel_pd