#encoding=GBK
import numpy as np
from tqdm import tqdm
from c_tfidf_calc import c_tf_idf , extract_n_words_perTopic , topic_size
from sklearn.metrics.pairwise import cosine_similarity
from data_process import data_process
import pandas as pd
from cluster import ump_sentence_res

docs = data_process()



#全局变量:--------------------------------------------------------------------------------------------
#构建docs_df DataFrame
docs_df = pd.DataFrame(docs ,columns = ['Doc'])
docs_df['Topic'] = ump_sentence_res['label']
docs_df['Doc_ID'] = range(len(docs_df))
# print(docs_df)

#全局变量:
#构建topic_size Dataframe
t_size = topic_size(docs_df)
# print(t_size)

#构建docs_per_topic DataFrame : 将文档按所属主题按文档组合成一篇大文档
docs_per_topic = docs_df.groupby(['Topic'] , as_index = False).agg({"Doc" : ' '.join})#将所有文档按主题分别连接为一个大文档
# print(docs_per_topic)

# print("number of topic: " , len(docs_per_topic['Doc'].tolist()))#话题数
# print(docs_per_topic['Doc'])

c_tfidf , count = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# tf矩阵
# print(c_tfidf)
# print("c_tfidf shape: " , c_tfidf.shape) #[len(word_dic) , topic_num]

# topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)

merge_Topic = t_size.loc[(t_size['Size'] < 100) & (t_size['Topic'] != -1)].Topic
merge_len = len(merge_Topic)
#---------------------------------------------------------------------------------------------------------

print("合并前主题的主题-文档分布:")
print(t_size)

def merge_topic(topic_size_min) :
    '''
    :param topic_size_min: 合并时的阈值，若小于这个值，将对该主题进行合并
    :return:
    '''
    global  c_tfidf , t_size , docs_df , docs_per_topic#全局变量声明

    print("需要合并的主题数量为:" , topic_size_min)

    for i in range(topic_size_min) :
        print("第" , i + 1 , "次合并主题")
        c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
        similarity = cosine_similarity(c_tfidf_transpose)#每个主题下单词分布两两之间的余弦相似度
        # print(similarity)
        np.fill_diagonal(similarity , 0) # 与自身的余弦相似度全部归0
        # print(similarity)
        # print(similarity.shape) # 16 * 16

        #确定需要合并的主题号，以及将其合并至的目标主题号
        topic_to_merge = t_size.iloc[-1].Topic # 需要合并的主题
        # print("topic_to_merge : " , topic_to_merge)
        topic_to_merge_in = np.argmax(similarity[topic_to_merge + 1]) - 1# -1 +1是因为该主题在similarity矩阵中的行列号 = 主题号 + 1
        # print("topic_to_merge_in : " , topic_to_merge_in)# 需要将该主题合并至的目标主题,即similarity矩阵中该主题行中最大值所对应的主题
        print("主题:" , topic_to_merge , "被合并到主题:" , topic_to_merge_in)

        #下面改变其主题
        docs_df.loc[docs_df['Topic'] == topic_to_merge , "Topic"] = topic_to_merge_in#在文档集docs_df中改变该文档所对应的主题
        #这一步只是将主题编号修改了，并未移除先前的主题序号

        #更新docs_df
        old_topics = docs_df.sort_values("Topic").Topic.unique()
        # print(old_topics)
        map_topic = {old_topic : index -1 for index , old_topic in enumerate(old_topics)}#将序号-1改设为现在的主题号
        # print(map_topic)
        docs_df.Topic = docs_df.Topic.map(map_topic) # index : topic
        # print(docs_df.sort_values("Topic").Topic.unique())

        #更新docs_per_topic
        docs_per_topic = docs_df.groupby(["Topic"] , as_index = False).agg({"Doc" : ' '.join})
        # print(docs_per_topic)

        #更新c_tf_idf矩阵,以及主题下对应的主题词
        c_tfidf , count = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# 更新tf矩阵
        # print(c_tfidf)

        #更新t_size
        t_size = topic_size(docs_df)
        print("第" , i + 1 ,"次合并后的主题分布")
        print(t_size)

        #更新similarity矩阵
        c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
        similarity = cosine_similarity(c_tfidf_transpose)  # 每个主题下单词分布两两之间的余弦相似度
        np.fill_diagonal(similarity, 0)  # 与自身的余弦相似度全部归0

        topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)
    return docs_df , t_size , docs_per_topic , c_tfidf , similarity , topic_words
