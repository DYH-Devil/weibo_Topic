#encoding=GBK
import pandas as pd
from data_process import DATA_PROCESS
from data_sort import WEIBO_CLASSIFY
from sentence_embedding import SENTENCE_EMBEDDING
from dimension_reduction import UMAP_REDUCTION_FORCLUSTER
from cluster import HDBSCAN_CLUSTER
from cluster_res_show import SHOW_CLUSTER

if __name__ == '__main__':
    #读取文件
    data_df = pd.read_csv('../data/weibo_data.csv')

    #分割数据集(热门微博，非热门微博)
    WEIBO_CLASSIFY(data_df , alpha = 0.5 , beta = 0.5 , theta = 0.5)

    hotopic_data = pd.read_csv('../data/hotopic.csv' , encoding = 'utf-8')#先对热门微博进行聚类

    #数据预处理
    data_process = DATA_PROCESS(hotopic_data)

    #句嵌入，将句子编码向量化
    sentence_embedding = SENTENCE_EMBEDDING(data_process)

    #数据降维(用于聚类分析，非可视化)
    umap_embedding_forcluster = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 25 , n_components = 10)

    #对降维后的数据进行聚类
    sentence_cluster = HDBSCAN_CLUSTER(min_cluster_size = 20)
    print(sentence_cluster.labels_)

    #聚类结果可视化
    #step1 将数据降为2维
    umap_embedding_forvisual = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 10 , n_components = 2)

    #step2 在图像上画出
    SHOW_CLUSTER(umap_embedding_forvisual)








