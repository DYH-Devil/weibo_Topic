#encoding=GBK

from  reduce_topic_num import merge_topic
from reduce_topic_num import merge_len
import joblib

docs_df , t_size , docs_per_topic , c_tfidf ,  similarity , topic_words = merge_topic(merge_len)


del topic_words[-1]#ȥ������
for topic , words in topic_words.items() :
    print(topic , words)


print("����ֲ�:")
print(t_size)

print("���ƶȾ���:")
print(similarity)
print(similarity.shape)

print("�ϲ������Ļ���-�ĵ���")
docs_per_topic = docs_per_topic.loc[docs_per_topic['Topic'] != -1]
print(docs_per_topic)