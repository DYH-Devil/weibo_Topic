#encoding=GBK

'''
���������н�ά
'''

import umap
import joblib
import os

def UMAP_REDUCTION_FORCLUSTER(sentence_embedding , n_nerghbors , n_components) :
    '''

    :param sentence_embedding: �����ľ�������
    :param n_nerghbors: umap����:�ٽ������
    :param n_components: ump����:�½�ά��
    :return: ��ά�������,���ھ���
    '''
    if (os.path.exists('../model/sentence_model.dat')):
        sentence_embedding = joblib.load('../model/sentence_model.dat')
    else:
        print("�Ҳ�����Ƕ��ģ��")

    model_name = '../model/umap_sentence_embedding' + str(n_components) + '.dat'
    if os.path.exists(model_name):
        umap_embedding = joblib.load(model_name)
    else:
        umap_model = umap.UMAP(
            n_neighbors=n_nerghbors,  # �ڽ������
            n_components=n_components,  # ��ά���ά��
            min_dist=0.0,
            metric='cosine',
            random_state=2022
        )
        umap_embedding = umap_model.fit_transform(sentence_embedding)
        joblib.dump(umap_embedding , model_name)

    return umap_embedding