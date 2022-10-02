import Bert_topic
import pandas as pd
from clean_function import text_process
from data_clean import word_split , del_stopwords

def doc_process(text) :

    text_clean = text_process(text)  # 清洗
    # for line in text_clean :
    #    print(line)
    # print(text_clean)
    text_split_list = word_split(text_clean)  # 分词
    # for line in text_split_list :
    #    print(line)
    text_del_stopwords = del_stopwords(text_split_list)  # 去停用词
    # for line in text_split_list :
    #    print(line)
    docs = []
    for text in text_del_stopwords :
        text = ''.join(text)
        docs.append(text)

    return docs


