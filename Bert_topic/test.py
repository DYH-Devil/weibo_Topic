#encoding=GBK
from data_clean import text_list as text
from data_process import doc_process
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
import jieba
from bertopic import BERTopic
import sentence_transformers.SentenceTransformer as SentenceTransformer
import joblib
import os

def tokenize_chinese(text) :
    words = jieba.lcut(text)
    return words

docs = doc_process(text)

chunk = [docs[i : i + 1000 ] for i in range(0 , len(docs) , 1000)]
print(len(chunk))
for index , line in enumerate(chunk[6]) :
    print(index , line)

sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')#句嵌入模型
if os.path.exists('./save_model/sentence_embedding.dat') :
    embeddings = joblib.load('./save_model/sentence_embedding.dat')

pca_model = IncrementalPCA(n_components = 5)
cluster_model = MiniBatchKMeans(n_clusters = 50 , random_state = 0)
vector_model = OnlineCountVectorizer(tokenizer = tokenize_chinese , stop_words = [' '])

topicmodel = BERTopic(
    embedding_model = sentence_model ,
    umap_model = pca_model ,
    hdbscan_model = cluster_model ,
    vectorizer_model = vector_model
)

topics = []
for index , doc in enumerate(chunk) :
    print("进行第" , index , "次")
    topicmodel.partial_fit(doc)
    topics.extend(topicmodel.topics_)
    print("当前热点话题")
    for i in range(5) :
        print(topicmodel.get_topic(i))
    print("-" * 50)





