from gensim.models import LdaModel , TfidfModel , CoherenceModel
from data_process.data_clean import word_split , del_stopwords
from data_process.clean_function import text_process
from data_process.word_cloud import WordCloud_process
from data_process.data_clean import text_list as text
from data_process.LDA import create_dict , LDA_topic
import matplotlib.pyplot as plt
import pyLDAvis.gensim

if __name__ == '__main__':
    text_clean = text_process(text)  # 清洗
    text_split_list = word_split(text_clean)  # 分词
    text_del_stopwords = del_stopwords(text_split_list)  # 去停用词

    #-----------------词云分析--------------------------------

    print("--------------------------词云分析-----------------------------")
    picture =  WordCloud_process(text_del_stopwords)
    plt.imshow(picture)
    plt.axis("off")
    plt.show()

    #-----------------LDA分析---------------------------------

    print("--------------------------LDA分析-----------------------------")
    texts = text_del_stopwords
    dictionary, corpus = create_dict(texts)
    # dictionary:根据语料生成的词典
    # corpus:[(id,time)]:(单词编号，文档中词语出现的次数)
    print(dictionary)
    # print(corpus)

    tfidf = TfidfModel(corpus)
    # tfidf:<num_docs=5777 , num_nnz=98313>
    # num_docs:表示corpus中的文档数 , num_nnz:表示每个文档中词数(不含重复(文档中不重复))之和
    # tfidf描述词语所具有的区分性，即哪些词是文档的关键词
    tfidf_corpus = tfidf[corpus]  # 将corpus中词的频数转为tfidf值[(id,tfidf),......]

    coherence_dict = {} #困惑度字典:{num_topic : coherence}

    # 话题数从1到5分别计算各自困惑度
    for i in range(1 , 6) :
        model = LDA_topic(texts = texts ,
                        corpus = tfidf_corpus ,
                        dictionary = dictionary ,
                        num_topics = i ,
                        num_words = 8)

        coherence_dict[i] = model.coherence()

    #-----------------最佳主题数---------------------------------

    best_topic_num = list(coherence_dict.values()).index(max(coherence_dict.values())) + 1
    print("The best topic num is : " , best_topic_num)
    # 最佳主题数为2

    # 列出最佳主题数下的关键词
    best_model = LDA_topic(texts = texts , 
                           corpus = tfidf_corpus , 
                           dictionary = dictionary , 
                           num_topics = best_topic_num , 
                           num_words = 8)
    topic_words = best_model.lda.print_topics(num_topics = best_topic_num , num_words = 10)
    print(topic_words)

    #-----------------可视化分析---------------------------------
    print("--------------------------pyLDAvis可视化分析-----------------------------")
    LDA_show = pyLDAvis.gensim.prepare(best_model.lda , corpus , dictionary)
    pyLDAvis.show(LDA_show)
    pyLDAvis.save_html(LDA_show , './picture/pyLDAvis_show.html')# 保存html文件