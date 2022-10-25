#encoding=GBK
'''
��ͼ����cluster�����Ľ��
'''

from matplotlib import pyplot as plt
import joblib
import os
import pandas as pd

def SHOW_CLUSTER(sentence_embedding) :
    '''
    :param sentence_embedding: ����������Ϊ2ά֮�������
    :return: None
    '''

    cluster_name = '../model/sentence_cluster.dat'
    if(os.path.exists(cluster_name)) :
        cluster = joblib.load(cluster_name)
    else :
        print("δ�ҵ�����������")

    cluster_res = pd.DataFrame(sentence_embedding , columns = ['X' , 'Y'])
    cluster_res['Topic'] = cluster.labels_

    fig, ax = plt.subplots(figsize=(25, 15))

    topic_cluster = cluster_res.loc[cluster_res['Topic'] != -1 , :]
    noise_cluster = cluster_res.loc[cluster_res['Topic'] == -1 , :]#��������

    #plt.scatter(noise_cluster['X'], noise_cluster['Y'], c='black', s=0.05, marker='x')
    plt.scatter(topic_cluster['X'], topic_cluster['Y'], c=topic_cluster['Topic'], s=0.05, cmap='hsv_r')

    plt.colorbar()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("������2άͶӰ")
    plt.show()
    plt.savefig('../res_save/cluster_pic.png')