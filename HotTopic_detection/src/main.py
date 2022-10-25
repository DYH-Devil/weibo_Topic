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

    # # 读取文件
    # data_df = pd.read_csv('../data/data.csv')
    #
    # # 分割数据集(热门微博，非热门微博)
    # WEIBO_CLASSIFY(data_df , alpha = 0.7 , beta = 0.2 , theta = 0.1)
    #

    # # 数据预处理
    # hotopic_data = pd.read_csv('../data/hotopic.csv' , encoding = 'utf-8')#先对热门微博进行聚类
    # data_process = DATA_PROCESS(hotopic_data)
    # # 保存预处理完成后的微博文档
    # data_process.to_csv('../data/data_process.csv')

    # 读取预处理后的数据
    data = pd.read_csv('../data/data_process.csv' , encoding = 'utf-8')
    sentence = data['微博正文'].to_list()

    word_spread_dict = joblib.load('../data/word_spread_dict.dat') # 加载词热度词典

    # 句嵌入，将句子编码向量化
    sentence_embedding = SENTENCE_EMBEDDING(sentence)

    #数据降维(用于聚类分析，非可视化)
    umap_embedding_forcluster = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 25 , n_components = 10)

    #对降维后的数据进行聚类
    sentence_cluster = HDBSCAN_CLUSTER(min_cluster_size = 50)
    # print(sentence_cluster.labels_)

    #聚类结果可视化
    #step1 将数据降为2维
    umap_embedding_forvisual = UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors = 10 , n_components = 2)

    #step2 在图像上画出
    SHOW_CLUSTER(umap_embedding_forvisual)
    #
    #输出热点话题下的关键词
    doc_df, topic_size, doc_per_topic, c_tfidf, topic_similarity, topic_words = MERGE_TOPIC(0.7)
    joblib.dump(topic_words , '../data/topic_words.dat')
    joblib.dump(topic_size , '../data/topic_size.dat')

    del topic_words[-1]
    for topic, words in topic_words.items():
        print(topic, words)

    print(topic_size)













