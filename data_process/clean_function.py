'''
数据清除
'''

import re
from string import punctuation as punctuation_string


def text_process(text_list) :
    text_clean = []
    for text in text_list :
        #-------先把英文标点转为中文标点------------
        text = text.replace("?", "？")
        text = text.replace(".", "。")
        text = text.replace(",", "，")
        #---------------------------------------

        filters = ['\t', '\n', '\x97', '\x96' , '$', '%', '&', '@' , '！', '，', '？', '——', '_', '。', '—', '…', '；', '“',
                   "”", '：','（','）','《','》', '、']
        text = re.sub('|'.join(filters), ' ', text)  # |表示综合所有的filters,全部替换
        text = re.sub("【.+?】", "", text)  # 去除【】里的内容，通常里面的内容是引用
        text = re.sub('http?://\S+|www\.\S+', '', text)  # 去除url链接
        text = re.sub("#.*?#", "", text)  # 去除话题引用
        text = re.sub("<.*?>", "", text)#去除html有关字符
        text = re.sub("\n", "", text)
        text = re.sub('[0-9]', '', text)#去除数字
        text = re.sub('[a-zA-Z]' , '' , text)#去除英文字符
        text = text.replace('*', '')
        text_clean.append(text)

    return text_clean
