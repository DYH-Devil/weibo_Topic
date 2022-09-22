from gensim.models import LdaModel , TfidfModel , CoherenceModel
from data_process.data_clean import word_split , del_stopwords
from data_process.clean_function import text_process
from data_process.word_cloud import WordCloud_process
from data_process.data_clean import text_list as text
from data_process.LDA import create_dict , LDA_topic
import matplotlib.pyplot as plt
from collections import defaultdict
import pyLDAvis.gensim

if __name__ == '__main__':
    text_clean = text_process(text)  # 清洗
    #for line in text_clean :
    #    print(line)
    # print(text_clean)
    text_split_list = word_split(text_clean)  # 分词
    #for line in text_split_list :
    #    print(line)
    text_del_stopwords = del_stopwords(text_split_list)  # 去停用词
    #for line in text_split_list :
    #    print(line)

    #-----------------词云分析--------------------------------

    #print("--------------------------词云分析-----------------------------")
    #picture =  WordCloud_process(text_del_stopwords)
    #plt.imshow(picture)
    #plt.axis("off")
    #plt.show()

    #-----------------LDA分析---------------------------------

    print("--------------------------LDA分析-----------------------------")
    texts = text_del_stopwords


    frequency = defaultdict(int)#词频字典
    for text in texts :
        for word in text :
            frequency[word] += 1
    #for (key ,value) in frequency.items() :
    #    print(key , value)
    

    dictionary, corpus = create_dict(texts)
    # dictionary:根据语料生成的词典
    # corpus:[(id,time)]:(单词编号，文档中词语出现的次数)
    print(len(dictionary))
    # print(corpus)
    
    tfidf = TfidfModel(corpus)
    # tfidf:<num_docs=5777 , num_nnz=98313>
    # num_docs:表示corpus中的文档数 , num_nnz:表示每个文档中词数(不含重复(文档中不重复))之和
    # tfidf描述词语所具有的区分性，即哪些词是文档的关键词
    tfidf_corpus = tfidf[corpus]  # 将corpus中词的频数转为tfidf值[(id,tfidf),......]


    # 列出最佳主题数下的关键词
    best_model = LDA_topic(texts = texts , 
                           corpus = tfidf_corpus , 
                           dictionary = dictionary , 
                           num_topics = 4 , 
                           num_words = 8)
    topic_words = best_model.lda.print_topics(num_topics = 4 , num_words = 8)
    for topic in topic_words :
        print(topic)

    ##-----------------可视化分析---------------------------------
    #print("--------------------------pyLDAvis可视化分析-----------------------------")
    #LDA_show = pyLDAvis.gensim.prepare(best_model.lda , corpus , dictionary)
    #pyLDAvis.show(LDA_show)
    #pyLDAvis.save_html(LDA_show , './picture/pyLDAvis_show.html')# 保存html文件