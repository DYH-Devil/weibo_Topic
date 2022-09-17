'''
该文件用于数据预处理--将元数据中的微博原文提取出来保存至content.csv文件
'''
import pandas as pd
import os

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

#-------------------------start---------------------------------------------

file = pd.read_csv('../data/data.csv' ,  encoding = 'utf-8')
# print(file.describe())
# print(file.head())


content = file['微博正文']
content.to_csv('../data/content.csv')

