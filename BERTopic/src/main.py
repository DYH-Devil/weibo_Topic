#encoding=GBK

from  reduce_topic_num import merge_topic
from reduce_topic_num import merge_len
import joblib

docs_df , t_size , docs_per_topic , c_tfidf ,  similarity , topic_words = merge_topic(merge_len)


del topic_words[-1]#去除噪声
for topic , words in topic_words.items() :
    print(topic , words)


print("主题分布:")
print(t_size)

print("相似度矩阵:")
print(similarity)
print(similarity.shape)

print("合并主题后的话题-文档集")
docs_per_topic = docs_per_topic.loc[docs_per_topic['Topic'] != -1]
print(docs_per_topic)