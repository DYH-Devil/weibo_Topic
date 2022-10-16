#encoding=GBK
'''
���ڼ���ÿ�������spreadϵ��
'''
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from data_process import DATA_PROCESS

#��ȡ�ļ�
hotopic_data = pd.read_csv('../data/hotopic.csv' , encoding = 'utf-8')
print(len(hotopic_data))
data_process = DATA_PROCESS(hotopic_data)#Ԥ����������ֻ��text
print(len(data_process))

stop_words = [i.strip() for i in open('../data/cn_stopwords.txt' , encoding = 'utf-8').readlines()]
count = CountVectorizer(ngram_range = (1 , 1) ,
                        stop_words = stop_words)

count.fit_transform(data_process)
VOCAB = count.vocabulary_.keys()

Spread_Dict = {}
for key in VOCAB :
    Spread_Dict[key] = 0 #��ʼ���ʵ�

# for key , value in Spread_Dict.items() :
#     print("key:" , key , "value:" , value)

text_spread = hotopic_data[['text','spread']]
print(text_spread)


