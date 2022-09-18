'''
清洗数据，去除无用字符字符
'''
# -*- coding: utf-8 -*-

import csv
import jieba
from copy import deepcopy

#--------------------读取文件-------------------------
file = open('./data/content.csv' , 'r' , encoding = 'utf-8')
reader = csv.reader(file)#按行读取csv文件
#reader:[['1','xxx'] , ['2','xxx']........]
#---------------------------------------------------


#--------------------制作语料------------------------
text_list = []
for row in reader :
    row = row[1]
    text_list.append(row)
text_list.pop(0)#去除head:微博正文(列头)'
# print(text_list)
#---------------------------------------------------
#分词
def word_split(text_clean) :
    '''
    :param text_clean: 清洗后的文本
    :return:
    '''
    text_split_list = []#分词后的总表
    for text in text_clean:
        text_split = jieba.lcut(text)#对每个句子进行分词
        text_split_copy = deepcopy(text_split)#remove前需要深拷贝，否则踩大坑
        for word in text_split_copy :
            if(word == ' ') :#删除空格
                text_split.remove(word)
        text_split_list.append(text_split)#text_split_list:[['x','x'...],['x','x'...]...]
    return text_split_list
#---------------------------------------------------------
# print(text_split_list)


#--------------------去停用词------------------------------
def del_stopwords(text_split_list) :
    '''
    :param text_split_list: 分词后的文本
    :return:
    '''
    stop_words = open('./data/my_stopwords.txt', 'r', encoding='utf-8').readlines()  # 读取停用词库
    stop_list = []  # 将文件中的停用词存入列表
    for i in stop_words:
        stop_list.append(i.strip())

    for text in text_split_list :
        text_copy = deepcopy(text)
        for word in text_copy :
            if(word in stop_list) :
                text.remove(word)
    return text_split_list




