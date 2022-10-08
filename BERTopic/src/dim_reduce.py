'''
对嵌入后的句向量进行降维
'''
import os.path

import umap
import joblib

if(os.path.exists('../model/sentence_model.dat')) :
    sentence_embedding = joblib.load('../model/sentence_model.dat')
else :
    print("找不到句嵌入模型")

if os.path.exists('../model/umap_embedding.dat') :
    umap_embedding = joblib.load('../model/umap_embedding.dat')
else :
    umap_embedding = umap.UMAP(
        n_neighbors = 25 ,#邻近点个数
        n_components = 10 ,#降维后的维数
        min_dist = 0.0 ,
        metric = 'cosine' ,
        random_state = 2022
    ).fit_transform(sentence_embedding)
    joblib.dump(umap_embedding , '../model/umap_embedding.dat')
        #TODO:参数理解

print(umap_embedding.shape)

#TODO:尝试其他文本降维方法


