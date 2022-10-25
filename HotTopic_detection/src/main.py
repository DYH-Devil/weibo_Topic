#encoding=GBK
import joblib
import pandas as pd
from data_process import DATA_PROCESS
from data_classify import WEIBO_CLASSIFY
from sentence_embedding import SENTENCE_EMBEDDING
from dimension_reduction import UMAP_REDUCTION_FORCLUSTER
from cluster import HDBSCAN_CLUSTER
from cluster_res_show import SHOW_CLUSTER
from contribute_var import CONTRIBUTE
from get_topn_words import EXTRACT_N_WORDS_PERTOPIC , C_TFIDF_CALCULATION
from merge_topic import MERGE_TOPIC

if __name__ == '__main__':

    # # ��ȡ�ļ�
    # data_df = pd.read_csv('../data/data.csv')
    #
    # # �ָ����ݼ�(����΢����������΢��)
    # WEIBO_CLASSIFY(data_df , alpha = 0.7 , beta = 0.2 , theta = 0.1)
    #

    # # ����Ԥ����
    # hotopic_data = pd.read_csv('../data/hotopic.csv' , encoding = 'utf-8')#�ȶ�����΢�����о���
    # data_process = DATA_PROCESS(hotopic_data)
    # # ����Ԥ������ɺ��΢���ĵ�
    # data_process.to_csv('../data/data_process.csv')

    # ��ȡԤ����������
    data = pd.read_csv('../data/data_process.csv' , encoding = 'utf-8')
    sentence = data['΢������'].to_list()

    word_spread_dict = joblib.load('../data/word_spread_dict.dat') # ���ش��ȶȴʵ�

    # ��Ƕ�룬�����ӱ���������
    sentence_embedding = SENTENCE_EMBEDDING(sentence)

    #���ݽ�ά(���ھ���������ǿ��ӻ�)
    umap_embedding_forcluster = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 25 , n_components = 10)

    #�Խ�ά������ݽ��о���
    sentence_cluster = HDBSCAN_CLUSTER(min_cluster_size = 50)
    # print(sentence_cluster.labels_)

    #���������ӻ�
    #step1 �����ݽ�Ϊ2ά
    umap_embedding_forvisual = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 10 , n_components = 2)

    #step2 ��ͼ���ϻ���
    SHOW_CLUSTER(umap_embedding_forvisual)
    #
    #����ȵ㻰���µĹؼ���
    doc_df, topic_size, doc_per_topic, c_tfidf, topic_similarity, topic_words = MERGE_TOPIC(0.7)
    joblib.dump(topic_words , '../data/topic_words.dat')
    joblib.dump(topic_size , '../data/topic_size.dat')

    del topic_words[-1]
    for topic, words in topic_words.items():
        print(topic, words)

    print(topic_size)













