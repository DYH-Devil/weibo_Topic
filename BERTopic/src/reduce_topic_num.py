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

c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
similarity = cosine_similarity(c_tfidf_transpose)  # 每个主题下单词分布两两之间的余弦相似度
np.fill_diagonal(similarity, 0)  # 与自身的余弦相似度全部归0

# topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)

merge_Topic = t_size.loc[(t_size['Size'] < 100) & (t_size['Topic'] != -1)].Topic
merge_len = len(merge_Topic)#需要合并的主题数目
#---------------------------------------------------------------------------------------------------------

print("合并前主题的主题-文档分布:")
print(t_size)

def adjust(topic_to_merge , topic_to_merge_in) :
    '''
    :param topic_to_merge: 需要调整的主题编号
    :param topic_to_merge_in: 将该主题调整至的目标主题编号
    :return:
    '''
    global c_tfidf, c_tfidf_transpose , t_size, docs_df, docs_per_topic , similarity  # 全局变量声明

    # 下面改变其主题
    docs_df.loc[docs_df['Topic'] == topic_to_merge, "Topic"] = topic_to_merge_in  # 在文档集docs_df中改变该文档所对应的主题
    # 这一步只是将主题编号修改了，并未移除先前的主题序号

    # 更新docs_df
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    # print(old_topics)
    map_topic = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}  # 将序号-1改设为现在的主题号
    # print(map_topic)
    docs_df.Topic = docs_df.Topic.map(map_topic)  # index : topic
    # print(docs_df.sort_values("Topic").Topic.unique())

    # 更新docs_per_topic
    docs_per_topic = docs_df.groupby(["Topic"], as_index=False).agg({"Doc": ' '.join})
    # print(docs_per_topic)

    # 更新c_tf_idf矩阵,以及主题下对应的主题词
    c_tfidf, count = c_tf_idf(docs_per_topic['Doc'], num_data=len(docs))  # 更新tf矩阵
    # print(c_tfidf)

    # 更新t_size
    t_size = topic_size(docs_df)
    # print("第" , i + 1 ,"次合并后的主题分布")
    # print(t_size)

    # 更新similarity矩阵
    c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
    similarity = cosine_similarity(c_tfidf_transpose)  # 每个主题下单词分布两两之间的余弦相似度
    np.fill_diagonal(similarity, 0)  # 与自身的余弦相似度全部归0
    print("合并后similarity shape")
    print(similarity.shape)


def merge_topic(topic_size_min , threshold = 0.7) :
    '''
    合并主题
    :param topic_size_min: 主题下文档数目的最小值，若不满足，则合并该主题
    :param threshold: 主题间的相似度阈值，若大于该阈值则合并两个主题 default = 0.7
    :return:
    '''
    print("开始合并主题......")
    print("-" * 50)

    print("第一步:合并文档数不足数的主题")

    # step1 : 按文档数量合并主题
    for i in range(topic_size_min) :
        print("第" , i + 1 , "次合并主题")

        print("合并前similarity shape")
        print(similarity.shape)

        # 确定需要合并的主题号，以及将其合并至的目标主题号
        topic_to_merge = t_size.iloc[-1].Topic  # 需要合并的主题
        # print("topic_to_merge : " , topic_to_merge)
        topic_to_merge_in = np.argmax(similarity[topic_to_merge + 1]) - 1  # -1 +1是因为该主题在similarity矩阵中的行列号 = 主题号 + 1
        # print("topic_to_merge_in : " , topic_to_merge_in)# 需要将该主题合并至的目标主题,即similarity矩阵中该主题行中最大值所对应的主题
        print("主题:" , topic_to_merge , "被合并到主题:" , topic_to_merge_in)
        print("-" * 50)

        adjust(topic_to_merge , topic_to_merge_in)# 调整全局变量

    print("第二步骤:合并主题相似度较大的主题")

    # step2 : 按主题相似度合并主题
    for row in range(similarity.shape[0] - 1) :#遍历similarity矩阵
        for column in range(similarity.shape[1] - 1) :
            if(similarity[row][column] >= threshold) :#若两个主题的相似度大于阈值，则认为它们属于一个主题
                topic_to_merge = row - 1
                topic_to_merge_in = column - 1
                adjust(topic_to_merge , topic_to_merge_in)# 调整全局变量

    topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)
    return docs_df , t_size , docs_per_topic , c_tfidf , similarity , topic_words
