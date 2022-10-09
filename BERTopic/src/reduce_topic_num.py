#encoding=GBK
import numpy as np
from c_tfidf_calc import c_tf_idf , extract_n_words_perTopic , topic_size
from sklearn.metrics.pairwise import cosine_similarity

def merge_topic(docs , c_tfidf , docs_df , t_size) :
    c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
    similarity = cosine_similarity(c_tfidf_transpose)#ÿ�������µ��ʷֲ�����֮����������ƶ�
    # print(similarity)
    np.fill_diagonal(similarity , 0) # ��������������ƶ�ȫ����0
    # print(similarity)
    # print(similarity.shape) # 16 * 16

    #ȷ����Ҫ�ϲ�������ţ��Լ�����ϲ�����Ŀ�������
    topic_to_merge = t_size.iloc[-1].Topic # ��Ҫ�ϲ�������
    # print("topic_to_merge : " , topic_to_merge)
    topic_to_merge_in = np.argmax(similarity[topic_to_merge + 1]) - 1# -1 +1����Ϊ��������similarity�����е����к� = ����� + 1
    # print("topic_to_merge_in : " , topic_to_merge_in)# ��Ҫ��������ϲ�����Ŀ������,��similarity�����и������������ֵ����Ӧ������

    #����ı�������
    docs_df.loc[docs_df['Topic'] == topic_to_merge , "Topic"] = topic_to_merge_in#���ĵ���docs_df�иı���ĵ�����Ӧ������
    #��һ��ֻ�ǽ��������޸��ˣ���δ�Ƴ���ǰ���������

    #�Ƴ�һ��������������������
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    # print(old_topics)
    map_topic = {old_topic : index -1 for index , old_topic in enumerate(old_topics)}#�����-1����Ϊ���ڵ������
    # print(map_topic)
    docs_df.Topic = docs_df.Topic.map(map_topic) # index : topic �޸�docs_df
    # print(docs_df.sort_values("Topic").Topic.unique())

    #�޸�docs_per_topic
    docs_per_topic = docs_df.groupby(["Topic"] , as_index = False).agg({"Doc" : ' '.join})
    # print(docs_per_topic)

    #���¼���c_tf_idfֵ,�Լ������¶�Ӧ�������
    c_tfidf_update , count_update = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# ����tf����
    print(c_tfidf_update)
    topic_words = extract_n_words_perTopic(c_tfidf_update , count_update , docs_per_topic , n_top = 10)
    return topic_words
