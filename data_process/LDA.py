'''lda主题模型进行话题识别'''
from gensim.models import LdaModel , TfidfModel , CoherenceModel
from gensim import corpora

def create_dict(texts):
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




