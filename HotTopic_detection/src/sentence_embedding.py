#encoding=GBK
'''
ʹ��SentenceBert�Ծ��ӽ���Ƕ��
'''

from sentence_transformers import SentenceTransformer
import joblib
import os

def SENTENCE_EMBEDDING(sentence_list) :
    '''
    :param sentence_list: Ԥ��������΢���ĵ�
    :return:
    '''
    SentenceModel = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')# ʹ��Ԥѵ��ģ��
    if os.path.exists('../model/sentence_model.dat'):
        sentence_embedding = joblib.load('../model/sentence_model.dat')

    else:
        sentence_embedding = SentenceModel.encode(sentence_list, show_progress_bar = True , device = "cuda:0")
        joblib.dump(sentence_embedding, '../model/sentence_model.dat')

    return sentence_embedding
