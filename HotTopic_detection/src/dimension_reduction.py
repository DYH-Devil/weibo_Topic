#encoding=GBK

'''
对向量进行降维
'''

import umap
import joblib
import os

def UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors , n_components) :
    '''

    :param sentence_embedding: 编码后的句子向量
    :param n_nerghbors: umap参数:临近点个数
    :param n_components: ump参数:下降维度
    :return: 降维后的向量,用于聚类
    '''
    if (os.path.exists('../model/sentence_model.dat')):
        sentence_embedding = joblib.load('../model/sentence_model.dat')
    else:
        print("找不到句嵌入模型")

    model_name = '../model/umap_sentence_embedding' + str(n_components) + '.dat'
    if os.path.exists(model_name):
        umap_embedding = joblib.load(model_name)
    else:
        umap_model = umap.UMAP(
            n_neighbors=n_nerghbors,  # 邻近点个数
            n_components=n_components,  # 降维后的维数
            min_dist=0.0,
            metric='cosine',
            random_state=2022
        )
        umap_embedding = umap_model.fit_transform(sentence_embedding)
        joblib.dump(umap_embedding , model_name)

    return umap_embedding