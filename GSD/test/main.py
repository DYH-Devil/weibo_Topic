import numpy
import numpy as np

from GSD.gsdmm.mgp import  MovieGroupProcess

np.random.seed(47)

file = open('../data/res_file.dat' , 'r' , encoding = 'utf-8')#输入数据暂时与BTM模型中的输入数据相同

texts = [line.split() for line in file.readlines()]
print(texts)

V = set()
for text in texts :
    for word in text :
        V.add(word)

doc_len = len(V)

print(doc_len)


mgp = MovieGroupProcess(K=500, n_iters=50, alpha=0.1, beta=0.1)
y = mgp.fit(texts, doc_len)


doc_cout = numpy.array(mgp.cluster_doc_count).argsort()[-mgp.cluster_count:][::-1]#根据每个主题下的文档数量排序,返回排序完成的下标:主题编号

def topwords(cluster_word_distribution , top_cluster , values) :
    save_file = open('../data/save_result.txt' , 'w' , encoding = 'utf-8')
    for cluster in top_cluster :
        top_words = sorted(cluster_word_distribution[cluster].items() , key = lambda k:k[1],reverse = True)[:values]#按主题下单词的出现次数对主题下的关键词排序
        print(top_words)
        save_file.write(str(top_words) + '\n')

topwords(mgp.cluster_word_distribution , doc_cout , 10)
