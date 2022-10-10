'''使用hdbscan对降维后的向量进行聚类'''
import os
import hdbscan
import joblib
import umap
import pandas as pd
from matplotlib import pyplot as plt

if(os.path.exists('../model/umap_embedding.dat')) :
    umap_embedding = joblib.load('../model/umap_embedding.dat')
else :
    print("找不到聚类所用的数据")
#在hdbscan聚类时使用的是15维的数据，在可视化时只用两维

if(os.path.exists('../model/hdbscan_res.dat')) :
    cluster = joblib.load('../model/hdbscan_res.dat')
else :
    cluster = hdbscan.HDBSCAN(
        min_cluster_size = 20 ,
        metric="euclidean",
        cluster_selection_method = 'eom' ,
        prediction_data = True
    ).fit(umap_embedding)
    joblib.dump(cluster , '../model/hdbscan_res.dat')


if(os.path.exists('../model/sentence_model.dat')) :
    sentence_embedding = joblib.load('../model/sentence_model.dat')
else :
    print("未指定用作降维可视化的数据!")


#对句子进行降维，用于可视化分析
if(os.path.exists('../model/sentence_Vis.dat')) :
    umap_sentence = joblib.load('../model/sentence_Vis.dat')
else :
    umap_sentence = umap.UMAP(
        n_neighbors = 10 ,#邻近点个数
        n_components = 2 ,#降维后的维数(2维)
        min_dist = 0.2 ,
        metric = 'cosine' ,
        random_state = 2022
    ).fit_transform(sentence_embedding)
    joblib.dump(umap_sentence , '../model/sentence_Vis.dat')

#print(umap_sentence.shape)
# (7600 , 2)

#可视化
ump_sentence_res = pd.DataFrame(umap_sentence , columns=['X' , 'Y'])
ump_sentence_res['label'] = cluster.labels_
#print(ump_sentence_res)

fig , ax = plt.subplots(figsize = (25 , 15))
clustered = ump_sentence_res.loc[ump_sentence_res['label'] != -1 , : ]
# print(clustered)

noise = ump_sentence_res.loc[ump_sentence_res['label'] == -1 , : ]
plt.scatter(noise['X'] , noise['Y'] , c = 'black' , s = 0.05 , marker = 'x')

plt.scatter(clustered['X'] , clustered['Y'] , c = clustered['label'] , s = 0.05 , cmap = 'hsv_r')
plt.colorbar()

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("聚类后的2维投影")
plt.show()
plt.savefig('../res_save/cluster_pic.png')


