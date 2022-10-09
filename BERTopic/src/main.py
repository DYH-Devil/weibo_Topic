#encoding=GBK

from data_process import data_process
import pandas as pd
from cluster import ump_sentence_res
from c_tfidf_calc import c_tf_idf , topic_size
from  reduce_topic_num import merge_topic

docs = data_process()

#构建docs_df DataFrame
docs_df = pd.DataFrame(docs ,columns = ['Doc'])
docs_df['Topic'] = ump_sentence_res['label']
docs_df['Doc_ID'] = range(len(docs_df))
print(docs_df)

#构建topic_size Dataframe
t_size = topic_size(docs_df)
# print(t_size)

#构建docs_per_topic DataFrame : 将文档按所属主题按文档组合成一篇大文档
docs_per_topic = docs_df.groupby(['Topic'] , as_index = False).agg({"Doc" : ' '.join})#将所有文档按主题分别连接为一个大文档
print(docs_per_topic)

# print("number of topic: " , len(docs_per_topic['Doc'].tolist()))#话题数
# print(docs_per_topic['Doc'])

c_tfidf , count = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# tf矩阵
# print(c_tfidf)
# print("c_tfidf shape: " , c_tfidf.shape) #[len(word_dic) , topic_num]

# topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)
topic_words = merge_topic(docs , c_tfidf , docs_df , t_size)
del topic_words[-1]#去除噪声
for topic , words in topic_words.items() :
    print(topic , words)