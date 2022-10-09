#encoding=GBK
import numpy as np
from c_tfidf_calc import c_tf_idf , extract_n_words_perTopic , topic_size
from sklearn.metrics.pairwise import cosine_similarity

def merge_topic(docs , c_tfidf , docs_df , t_size) :
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

    #下面改变其主题
    docs_df.loc[docs_df['Topic'] == topic_to_merge , "Topic"] = topic_to_merge_in#在文档集docs_df中改变该文档所对应的主题
    #这一步只是将主题编号修改了，并未移除先前的主题序号

    #移除一个主题后重新设置主题号
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    # print(old_topics)
    map_topic = {old_topic : index -1 for index , old_topic in enumerate(old_topics)}#将序号-1改设为现在的主题号
    # print(map_topic)
    docs_df.Topic = docs_df.Topic.map(map_topic) # index : topic 修改docs_df
    # print(docs_df.sort_values("Topic").Topic.unique())

    #修改docs_per_topic
    docs_per_topic = docs_df.groupby(["Topic"] , as_index = False).agg({"Doc" : ' '.join})
    # print(docs_per_topic)

    #重新计算c_tf_idf值,以及主题下对应的主题词
    c_tfidf_update , count_update = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# 更新tf矩阵
    print(c_tfidf_update)
    topic_words = extract_n_words_perTopic(c_tfidf_update , count_update , docs_per_topic , n_top = 10)
    return topic_words
