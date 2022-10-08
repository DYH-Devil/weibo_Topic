'''
句子嵌入
'''
import os.path

from data_process import data_process
from sentence_transformers import SentenceTransformer
import joblib

texts = data_process()
print(texts)

sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')#使用预训练模型
# TODO:尝试不一样的预训练模型

if os.path.exists('../model/sentence_model.dat') :
    sentence_embedding = joblib.load('../model/sentence_model.dat')

else :
    sentence_embedding = sentence_model.encode(texts , show_progress_bar = True)
    joblib.dump(sentence_embedding , '../model/sentence_model.dat')

print(sentence_embedding.shape)

