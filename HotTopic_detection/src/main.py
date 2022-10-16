#encoding=GBK
import pandas as pd
from data_process import DATA_PROCESS
from data_sort import WEIBO_CLASSIFY
from sentence_embedding import SENTENCE_EMBEDDING
from dimension_reduction import UMAP_REDUCTION_FORCLUSTER
from cluster import HDBSCAN_CLUSTER
from cluster_res_show import SHOW_CLUSTER

if __name__ == '__main__':
    #��ȡ�ļ�
    data_df = pd.read_csv('../data/weibo_data.csv')

    #�ָ����ݼ�(����΢����������΢��)
    WEIBO_CLASSIFY(data_df , alpha = 0.5 , beta = 0.5 , theta = 0.5)

    hotopic_data = pd.read_csv('../data/hotopic.csv' , encoding = 'utf-8')#�ȶ�����΢�����о���

    #����Ԥ����
    data_process = DATA_PROCESS(hotopic_data)

    #��Ƕ�룬�����ӱ���������
    sentence_embedding = SENTENCE_EMBEDDING(data_process)

    #���ݽ�ά(���ھ���������ǿ��ӻ�)
    umap_embedding_forcluster = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 25 , n_components = 10)

    #�Խ�ά������ݽ��о���
    sentence_cluster = HDBSCAN_CLUSTER(min_cluster_size = 20)
    print(sentence_cluster.labels_)

    #���������ӻ�
    #step1 �����ݽ�Ϊ2ά
    umap_embedding_forvisual = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 10 , n_components = 2)

    #step2 ��ͼ���ϻ���
    SHOW_CLUSTER(umap_embedding_forvisual)








