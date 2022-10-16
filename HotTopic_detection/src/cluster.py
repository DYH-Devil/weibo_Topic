#encoding=GBK
'''
ʹ��hdbscan�Խ�ά������ݽ��о������
'''

import hdbscan
import joblib
import os

def HDBSCAN_CLUSTER(min_cluster_size) :
    '''
    :param min_cluster_size: һ����������������Сֵ
    :return:
    '''
    ump_model_name = '../model/umap_sentence_embedding10.dat'
    if (os.path.exists(ump_model_name)):
        umap_sentence_embedding = joblib.load(ump_model_name)
    else:
        print("�Ҳ����������õ�����")
    # ��hdbscan����ʱʹ�õ���15ά�����ݣ��ڿ��ӻ�ʱֻ����ά

    if (os.path.exists('../model/hdbscan_res.dat')):
        sentence_cluster = joblib.load('../model/hdbscan_res.dat')
    else:
        cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method='eom',
            prediction_data=True
        )
        sentence_cluster = cluster_model.fit(umap_sentence_embedding)
        joblib.dump(sentence_cluster , '../model/sentence_cluster.dat')

    return sentence_cluster
