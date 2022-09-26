'''lda主题模型进行话题识别'''
from gensim.models import LdaModel , TfidfModel , CoherenceModel
from gensim import corpora
from collections import defaultdict

def create_dict(texts):
    '''
    :param texts: 语料
    根据语料创建词典
    '''
    frequency = defaultdict(int)#词频字典
    for text in texts :
        for word in text :
            frequency[word] += 1
    

    dictionary = corpora.Dictionary(texts)

    bad_ids = []
    for (key , value) in frequency.items() :
        if(value < 50) :
            bad_ids.append(dictionary.token2id[key])

    dictionary.filter_tokens(bad_ids = bad_ids)#去除dictionary中词频小于10的词语
    dictionary.compactify()
    dictionary.save('./data/my_dictionary.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary , corpus


class LDA_topic() :
    def __init__(self , texts , corpus , dictionary , num_topics , num_words):
        self.texts = texts
        self.corpus = corpus
        self.dictionary = dictionary
        self.num_topics = num_topics
        self.num_words = num_words
        self.lda = LdaModel(corpus = self.corpus ,
                            id2word = self.dictionary ,
                            num_topics = self.num_topics ,
                            passes = 30 ,
                            # chunksize = 1000 ,
                            random_state = 10 , 
                            alpha = 0.25 ,
                            eta = 0.01 , 
                            iterations = 100
                            )

    def coherence(self):
        model = self.lda
        ldacm = CoherenceModel(model = model , corpus = self.corpus , dictionary = self.dictionary , coherence = 'u_mass')
        return ldacm.get_coherence()




