#encoding=GBK
'''
�Ի����ȶ�����
'''
import joblib
import os

import pandas as pd

if(os.path.exists('../data/topic_words.dat')) :
    topic_words = joblib.load('../data/topic_words.dat')

if(os.path.exists('../data/topic_size.dat')) :
    topic_size = joblib.load('../data/topic_size.dat')


topic_heat_list = []
topic_list = []

del topic_words[-1]#��Ҫ����
for topic , words in topic_words.items() :
    print(topic , words)

for topic , words in topic_words.items() :
    topic_heat_sum = 0
    for item in words :
        topic_heat_sum += item[1] #topic���������еĴ����ȶ�*c-tfidf֮��
    topic_heat_avg  = topic_heat_sum / topic_size.loc[topic_size['Topic'] == topic].Size.values[0]
    topic_heat_list.append(topic_heat_avg)#��һ�������ƽ���ĵ��ȶ�
    topic_list.append(topic)

topic_heat_df = pd.DataFrame({
    'Topic':topic_list,
    'Topic_heat':topic_heat_list
})

print(topic_heat_df.sort_values(by = 'Topic_heat' , ascending = False))

