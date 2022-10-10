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

c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
similarity = cosine_similarity(c_tfidf_transpose)  # ÿ�������µ��ʷֲ�����֮����������ƶ�
np.fill_diagonal(similarity, 0)  # ��������������ƶ�ȫ����0

# topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)

merge_Topic = t_size.loc[(t_size['Size'] < 100) & (t_size['Topic'] != -1)].Topic
merge_len = len(merge_Topic)#��Ҫ�ϲ���������Ŀ
#---------------------------------------------------------------------------------------------------------

print("�ϲ�ǰ���������-�ĵ��ֲ�:")
print(t_size)

def adjust(topic_to_merge , topic_to_merge_in) :
    '''
    :param topic_to_merge: ��Ҫ������������
    :param topic_to_merge_in: ���������������Ŀ��������
    :return:
    '''
    global c_tfidf, c_tfidf_transpose , t_size, docs_df, docs_per_topic , similarity  # ȫ�ֱ�������

    # ����ı�������
    docs_df.loc[docs_df['Topic'] == topic_to_merge, "Topic"] = topic_to_merge_in  # ���ĵ���docs_df�иı���ĵ�����Ӧ������
    # ��һ��ֻ�ǽ��������޸��ˣ���δ�Ƴ���ǰ���������

    # ����docs_df
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    # print(old_topics)
    map_topic = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}  # �����-1����Ϊ���ڵ������
    # print(map_topic)
    docs_df.Topic = docs_df.Topic.map(map_topic)  # index : topic
    # print(docs_df.sort_values("Topic").Topic.unique())

    # ����docs_per_topic
    docs_per_topic = docs_df.groupby(["Topic"], as_index=False).agg({"Doc": ' '.join})
    # print(docs_per_topic)

    # ����c_tf_idf����,�Լ������¶�Ӧ�������
    c_tfidf, count = c_tf_idf(docs_per_topic['Doc'], num_data=len(docs))  # ����tf����
    # print(c_tfidf)

    # ����t_size
    t_size = topic_size(docs_df)
    # print("��" , i + 1 ,"�κϲ��������ֲ�")
    # print(t_size)

    # ����similarity����
    c_tfidf_transpose = c_tfidf.T  # shape[topic_num , len(word_dic)]
    similarity = cosine_similarity(c_tfidf_transpose)  # ÿ�������µ��ʷֲ�����֮����������ƶ�
    np.fill_diagonal(similarity, 0)  # ��������������ƶ�ȫ����0
    print("�ϲ���similarity shape")
    print(similarity.shape)


def merge_topic(topic_size_min , threshold = 0.7) :
    '''
    �ϲ�����
    :param topic_size_min: �������ĵ���Ŀ����Сֵ���������㣬��ϲ�������
    :param threshold: ���������ƶ���ֵ�������ڸ���ֵ��ϲ��������� default = 0.7
    :return:
    '''
    print("��ʼ�ϲ�����......")
    print("-" * 50)

    print("��һ��:�ϲ��ĵ���������������")

    # step1 : ���ĵ������ϲ�����
    for i in range(topic_size_min) :
        print("��" , i + 1 , "�κϲ�����")

        print("�ϲ�ǰsimilarity shape")
        print(similarity.shape)

        # ȷ����Ҫ�ϲ�������ţ��Լ�����ϲ�����Ŀ�������
        topic_to_merge = t_size.iloc[-1].Topic  # ��Ҫ�ϲ�������
        # print("topic_to_merge : " , topic_to_merge)
        topic_to_merge_in = np.argmax(similarity[topic_to_merge + 1]) - 1  # -1 +1����Ϊ��������similarity�����е����к� = ����� + 1
        # print("topic_to_merge_in : " , topic_to_merge_in)# ��Ҫ��������ϲ�����Ŀ������,��similarity�����и������������ֵ����Ӧ������
        print("����:" , topic_to_merge , "���ϲ�������:" , topic_to_merge_in)
        print("-" * 50)

        adjust(topic_to_merge , topic_to_merge_in)# ����ȫ�ֱ���

    print("�ڶ�����:�ϲ��������ƶȽϴ������")

    # step2 : ���������ƶȺϲ�����
    for row in range(similarity.shape[0] - 1) :#����similarity����
        for column in range(similarity.shape[1] - 1) :
            if(similarity[row][column] >= threshold) :#��������������ƶȴ�����ֵ������Ϊ��������һ������
                topic_to_merge = row - 1
                topic_to_merge_in = column - 1
                adjust(topic_to_merge , topic_to_merge_in)# ����ȫ�ֱ���

    topic_words = extract_n_words_perTopic(c_tfidf , count , docs_per_topic , n_top = 10)
    return docs_df , t_size , docs_per_topic , c_tfidf , similarity , topic_words
