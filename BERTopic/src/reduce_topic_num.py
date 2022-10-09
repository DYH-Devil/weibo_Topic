#encoding=GBK
import numpy as np
from tqdm import tqdm
from c_tfidf_calc import c_tf_idf , extract_n_words_perTopic , topic_size
from sklearn.metrics.pairwise import cosine_similarity
from data_process import data_process
import pandas as pd
from cluster import ump_sentence_res

docs = data_process()



#ȫ�ֱ���:--------------------------------------------------------------------------------------------
#����docs_df DataFrame
docs_df = pd.DataFrame(docs ,columns = ['Doc'])
docs_df['Topic'] = ump_sentence_res['label']
docs_df['Doc_ID'] = range(len(docs_df))
# print(docs_df)

#ȫ�ֱ���:
#����topic_size Dataframe
t_size = topic_size(docs_df)
# print(t_size)

#����docs_per_topic DataFrame : ���ĵ����������ⰴ�ĵ���ϳ�һƪ���ĵ�
docs_per_topic = docs_df.groupby(['Topic'] , as_index = False).agg({"Doc" : ' '.join})#�������ĵ�������ֱ�����Ϊһ�����ĵ�
# print(docs_per_topic)

# print("number of topic: " , len(docs_per_topic['Doc'].tolist()))#������
# print(docs_per_topic['Doc'])

c_tfidf , count = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# tf����
# print(c_tfidf)
# print("c_tfidf shape: " , c_tfidf.shape) #[len(word_dic) , topic_num]

# topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)

merge_Topic = t_size.loc[(t_size['Size'] < 100) & (t_size['Topic'] != -1)].Topic
merge_len = len(merge_Topic)
#---------------------------------------------------------------------------------------------------------

print("�ϲ�ǰ���������-�ĵ��ֲ�:")
print(t_size)

def merge_topic(topic_size_min) :
    '''
    :param topic_size_min: �ϲ�ʱ����ֵ����С�����ֵ�����Ը�������кϲ�
    :return:
    '''
    global  c_tfidf , t_size , docs_df , docs_per_topic#ȫ�ֱ�������

    print("��Ҫ�ϲ�����������Ϊ:" , topic_size_min)

    for i in range(topic_size_min) :
        print("��" , i + 1 , "�κϲ�����")
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
        print("����:" , topic_to_merge , "���ϲ�������:" , topic_to_merge_in)

        #����ı�������
        docs_df.loc[docs_df['Topic'] == topic_to_merge , "Topic"] = topic_to_merge_in#���ĵ���docs_df�иı���ĵ�����Ӧ������
        #��һ��ֻ�ǽ��������޸��ˣ���δ�Ƴ���ǰ���������

        #����docs_df
        old_topics = docs_df.sort_values("Topic").Topic.unique()
        # print(old_topics)
        map_topic = {old_topic : index -1 for index , old_topic in enumerate(old_topics)}#�����-1����Ϊ���ڵ������
        # print(map_topic)
        docs_df.Topic = docs_df.Topic.map(map_topic) # index : topic
        # print(docs_df.sort_values("Topic").Topic.unique())

        #����docs_per_topic
        docs_per_topic = docs_df.groupby(["Topic"] , as_index = False).agg({"Doc" : ' '.join})
        # print(docs_per_topic)

        #����c_tf_idf����,�Լ������¶�Ӧ�������
        c_tfidf , count = c_tf_idf(docs_per_topic['Doc'] , num_data = len(docs))# ����tf����
        # print(c_tfidf)

        #����t_size
        t_size = topic_size(docs_df)
        print("��" , i + 1 ,"�κϲ��������ֲ�")
        print(t_size)

        #����similarity����
        c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
        similarity = cosine_similarity(c_tfidf_transpose)  # ÿ�������µ��ʷֲ�����֮����������ƶ�
        np.fill_diagonal(similarity, 0)  # ��������������ƶ�ȫ����0

        topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)
    return docs_df , t_size , docs_per_topic , c_tfidf , similarity , topic_words
