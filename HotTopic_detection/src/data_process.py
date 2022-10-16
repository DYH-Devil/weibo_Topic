#encoding=GBK

'''
数据预处理
包括清洗，分词，去停用
'''

import re
import jieba

#数据清洗
def DATA_CLEAN(text_list) :
    text_clean = []
    for text in text_list :
        #-------先把英文标点转为中文标点------------
        text = text.replace("?", "？")
        text = text.replace(".", "。")
        text = text.replace(",", "，")
        text = text.replace("[" , "{")
        text = text.replace("]" , "}")
        #---------------------------------------

        filters = ['\t', '\n', '\x97', '\x96' , '$', '%', '&', '@' , '！', '，', '？', '——', '_', '。', '—', '…', '；', '“','-',
                   "”", '：','（','）','《','》', '、', '·','~','(',')','!',':','/','＃','─']
        text = re.sub("{.+?}","",text)
        text = re.sub('|'.join(filters), '', text)  # |表示综合所有的filters,全部替换
        text = re.sub("【.+?】", "", text)  # 去除【】里的内容，通常里面的内容是引用
        text = re.sub('http?://\S+|www\.\S+', '', text)  # 去除url链接
        text = re.sub("#.*?#", "", text)  # 去除话题引用
        text = re.sub("<.*?>", "", text)#去除html有关字符
        text = re.sub("\n", "", text)
        text = re.sub('[0-9]', '', text)#去除数字
        text = re.sub('[a-zA-Z]' , '' , text)#去除英文字符
        text = text.replace('*', '')
        text = text.replace(" " , "")
        if(len(text) > 4) :
            text_clean.append(text)

    return text_clean

#分词
def TEXT_SPLIT(text_list) :
    text_split = []
    for text in text_list :
        text = jieba.lcut(text)
        text_split.append(text)
    return text_split

#去停用词
def DEL_STOPWORDS(text_list) :
    stop_file = open('../data/cn_stopwords.txt' , 'r' , encoding = 'utf-8')
    stop_words = []
    for word in stop_file.readlines():#加载停用词
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
    :param data_Df: 微博文档集 DataFrame
    :return: 预处理完的微博数据集
    '''
    data_df['text'] = data_df['text'].astype(str)
    text = data_df['text'].to_list()
    text_clean = DATA_CLEAN(text)
    text_split = TEXT_SPLIT(text_clean)
    text_del_stopwords = DEL_STOPWORDS(text_split)
    return text_del_stopwords

