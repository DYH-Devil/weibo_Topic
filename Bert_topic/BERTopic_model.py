import os.path

import pandas as pd

from data_process import doc_process
from clean_function import text_process
from data_clean import text_list as text
import sentence_transformers.SentenceTransformer as SentenceTransformer
from bertopic import BERTopic
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import jieba

#重新定义分词方式,使用中文分词
def tokenize_chinese(text) :
    words = jieba.lcut(text)
    return words
vectorizer = CountVectorizer(tokenizer = tokenize_chinese)


docs = doc_process(text)
print(docs)

#odcs:['w1 w2 w3','w4 w5 w6'......]

sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')#句嵌入模型

if os.path.exists('./save_model/sentence_embedding.dat') :
    embeddings = joblib.load('./save_model/sentence_embedding.dat')
else :
    embeddings = sentence_model.encode(docs , show_progress_bar = True)
    joblib.dump(embeddings , './save_model/sentence_embedding.dat')

print(embeddings.shape)

embedding_df = pd.DataFrame(embeddings)

model = BERTopic(embedding_model = sentence_model , verbose = True , vectorizer_model = vectorizer)
topics , probabilities = model.fit_transform(docs[:1000] , embeddings = embedding_df.iloc[:1000].values)
#每个文档对应的主题

for i in range(4) :
    print(model.get_topic(i))


