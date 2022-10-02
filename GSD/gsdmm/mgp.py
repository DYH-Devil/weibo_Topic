from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax
import json

class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual GSDMM introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.

        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        '''
        self.K = K # 主题上限数
        self.alpha = alpha #alpha:[0,1]与mgp中规则1有关,文档(学生)会选择文档(学生)数量更多的主题(table)
        self.beta = beta #beta:[0,1]与mgp中规则2有关，文档(学生)会选择与自己文档更相似(电影)的主题(table)
        self.n_iters = n_iters #迭代次数

        # 初始化参数
        self.number_docs = None #文档数量
        self.vocab_size = None #单词数量:向量长度V,每个单词表示为V中的一维分量
        self.cluster_doc_count = [0 for _ in range(K)] #list:K个主题下的文档数量，初始全为0
        self.cluster_word_count = [0 for _ in range(K)] #list:K个主题下的单词数量，初始全为0
        self.cluster_word_distribution = [{} for i in range(K)] #dict:K个dict，记录每个单词w在主题下的出现次数,单词分布

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        '''
        Reconstitute a MovieGroupProcess from previously fit data
        从data中获取以上参数，参数意义与上同
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        '''
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        '''
        随机采样
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        # m_z:主题下的文档数
        # n_z:主题下的单词数
        # n_w_z:单词w在主题z下的出现次数

        self.cluster_count = K
        d_z = [None for i in range(len(docs))] #list:表示每个文档的主题标记

        # 先对各个文档进行初始化
        for i, doc in enumerate(docs):

            # 为每个文档先随机分配一个主题
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z #标记各个文档的主题
            m_z[z] += 1 #对应主题下的文档数+1
            n_z[z] += len(doc) #对应主题下的单词数+len(doc)

            for word in doc:#对于每个单词
                if word not in n_z_w[z]: #根据每个单词在该主题下的出现次数进行计数
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        #开始迭代
        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # 先将该文档移出所对应的主题
                z_old = d_z[i]# 记录移出前的主题编号(旧的主题编号)

                m_z[z_old] -= 1 #旧的主题下文档-1
                n_z[z_old] -= len(doc) #旧的主题下单词数-len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1 #旧的主题下该文档的每个单词出现次数-1

                    # 若单词出现次数为0则将这个单词移除
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)#采样的是所计算score最大值所对应的主题编号

                # 若文档的主题发生改变
                if z_new != z_old:
                    total_transfers += 1#total_transfers记录的是每个文档在迭代中改变主题的次数

                #--操作与上类似，对每个参数进行更新--
                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1
                #----------------------------

            cluster_count_new = sum([1 for v in m_z if v > 0])# 新的主题数
            print("In stage %d: transferred %d clusters with %d clusters populated" % (
            _iter, total_transfers, cluster_count_new))
            if total_transfers == 0 and cluster_count_new == self.cluster_count:
                #一趟迭代中主题变更次数为0    主题数量不再变化    迭代次数超过25次 均满足 才停止
                print("Converged.  Breaking out.")
                break
            self.cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z# 返回的是每个文档的主题编号

    def score(self, doc):
        '''
        Score a document

        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size +1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return [pp/pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p),max(p)
