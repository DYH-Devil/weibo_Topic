'''lda主题模型进行话题识别'''
from gensim.models import LdaModel , TfidfModel , CoherenceModel
from gensim import corpora

def create_create_dict(texts):
    '''
    :param texts: 语料
    根据语料创建词典
    '''
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary , corpus


class LDA_topic() :
    def __init__(self , texts , corpus , dictionary , num_topics , num_words):
        self.texts = texts
        self.corpus = corpus
        self.dictionary = dictionary
        self.num_topics = num_topics
        self.num_words = num_words
        self.lda = LdaModel(corpus = self.corpus , id2word = self.dictionary , num_topics = num_topics , passes = 10 , random_state = 1)

    def coherence(self):
        model = self.lda
        ldacm = CoherenceModel(model = model , corpus = self.corpus , dictionary = self.dictionary , coherence = 'u_mass')
        return ldacm.get_coherence()


# if __name__ == '__main__':
#     text_clean = text_process(text)  # 清洗
#     text_split_list = word_split(text_clean)  # 分词
#     text_del_stopwords = del_stopwords(text_split_list)  # 去停用词
#
#     #-----------------数据处理完毕---------------------------------
#
#     texts = text_del_stopwords
#     dictionary, corpus = create_create_dict(texts)
#     # dictionary:根据语料生成的词典
#     # corpus:[(id,time)]:(单词编号，次数)
#     print(dictionary)
#     # print(corpus)
#
#     tfidf = TfidfModel(corpus)
#     # tfidf:<num_docs=5777 , num_nnz=98313>
#     # num_docs:表示corpus中的文档数 , num_nnz:表示每个文档中词数(不含重复)之和
#     # tfidf描述词语所具有的区分性，即哪些词是文档的关键词
#     tfidf_corpus = tfidf[corpus]  # 将corpus中词的频数转为tfidf值[(id,tfidf),......]
#
#     coherence_list = []
#     for i in range(1 , 6) :
#         model = LDA_topic(texts = texts ,
#                         corpus = tfidf_corpus ,
#                         dictionary = dictionary ,
#                         num_topics = i ,
#                         num_words = 8)
#
#         print('The topic_num is: ' , i ,
#               'topic words is: ' , model.lda.print_topics(num_topics = i , num_words = model.num_words) ,
#               'coherence is: ' , model.coherence())
#
#         coherence_list.append(model.coherence())




