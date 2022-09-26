# -*- coding: utf-8 -*-
from pvec import *
import numpy as np
from doc import Doc
from sampler import *


class Model():
    '''
    @description: 函数的功能是生成模型
    @param
    @return:
    '''
    W = 0  # vocabulary size:词典长度
    K = 0  # number of topics:话题数
    n_iter = 0  # maximum number of iteration of Gibbs Sampling，吉布斯采样的最大迭代次数。
    save_step = 0
    alpha = 0  # hyperparameters of p(z)
    beta = 0  # hyperparameters of p(w|z)
    nb_z = Pvec()  # n(b|z), size K*1 所有的词对biterm分配到主题z上的次数,在论文中是用来计算Nz的。今天必须把这个代码看懂 2020-03-03
    nwz = np.zeros((1, 1))  # n(w,z), size K*W，表示的是单词分配到主题z上的次数.
    pw_b = Pvec()  # the background word distribution。这三个参数都是为了计算每个主题中单词的分布而存在的。
    bs = []

    '''
        If true, the topic 0 is set to a background topic that 
        equals to the empirical word distribution. It can filter
        out common words
    '''
    has_background = False

    def __init__(self, K, W, a, b, n_iter, save_step, has_b=False):
        self.K = K
        self.W = W
        self.alpha = a
        self.beta = b
        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = has_b
        self.pw_b.resize(W)
        self.nwz.resize((K, W) , refcheck = False)
        self.nb_z.resize(K)

    def run(self, doc_pt, res_dir):
        '''
        @description: 生成模型运行函数，狄利克雷-多项 共轭分布，Gibbs采样。学明白就买AJ（1000+）或者买椅子（300左右）
        @param {type}
        @return:
        '''
        self.load_docs(doc_pt)  # 生成self.pw_b 和 self.bs。 目前self.pw_b表示的是每个单词对应的词频，一共有7个单词，那么pw_b的size就是7，self.bs表示的是所有的biterm
        # print('==========')
        # print(self.pw_b.size())
        # for item in self.bs:
        #     print(item.get_wi(), item.get_wj())
        # print('==========')
        # 以上代码是用来初始化self.pw_b 和 self.bs

        #self.bs:语料库中的所有词对
        #self.pw_b:语料库中的词频率


        self.model_init()  # 初始化 self.nb_z 和 self.nwz  #self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。#self.nwz[2][3] 表示的是在出题2中，2号单词出现的次数。
        # print('============================')
        print(self.nwz)  # 二维数组: 主题下的单词次数
        #print(self.nb_z[1])  # 词对库中主题的出现次数
        # print('=======================')
        # print('\n')
        print("Begin iteration")
        out_dir = res_dir + "k" + str(self.K) + "."
        for i in range(1, self.n_iter + 1):
            print("\riter " + str(i) + "/" + str(self.n_iter), end='\r')
            for b in range(len(self.bs)):
                # 根据每个biterm更新文章中的参数。
                self.update_biterm(self.bs[b])  # 计算核心代码，self.bs中保存的是词对的biterm类，代码是对每一个词对进行更新的。
            if i % self.save_step == 0:
                self.save_res(out_dir)
            print("==============================已迭代完第" , i , "次===================================")

        self.save_res(out_dir)


    def model_init(self):
        '''
        @description: 初始化模型的代码。
        @param :None
        @return: 生成self.nb_z 和self.nwz，
        @self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。z主题的多项分布 self.nb_z[1]:表示的第一个主题出现的次数。
        @self.nw_z: 表示主题下的单词数 , self.nwz[2][3] 表示的是在主题2中，2号单词出现的次数。
        '''
        for biterm in self.bs:  # 为每个词对分配主题
            # print('==========')
            # print(biterm.get_wi(), biterm.get_wj())
            k = uni_sample(self.K)  # k表示的是从0-K之间的随机数。用来初始化
            # print(k)
            self.assign_biterm_topic(biterm, k)  # 入参是一个词对和他对应的主题
            # print('============')
            print('\n')
        #遍历完每个词对(为每个词对分配主题后)便统计出了self.nb_z和self.nwz

    def load_docs(self, docs_pt):
        '''
        @description: 统计文档中的词频self.pw_b以及组成的词对self.bs,初始化
        @param docs_pt:
        @return:
        '''
        print("load docs: " + docs_pt)
        rf = open(docs_pt)
        if not rf:
            print("file not found: " + docs_pt)

        for line in rf.readlines():
            d = Doc(line)  # 初始化，将数字序列存入d.ws列表中

            biterms = []  # 一句话里的单词能组成的词对。
            d.gen_biterms(biterms)  # 产生词对,存入biterms
            # statistic the empirical word distribution
            for i in range(d.size()):
                w = d.get_w(i)
                self.pw_b[w] += 1  # 这行代码是在统计文本中的词频
            for b in biterms:  # b表示一个封装成类的词对
                self.bs.append(b)  # self.bs中添加的是一个biterm类。类的内容是这段文本中所有可能的词的组合.
        self.pw_b.normalize()  # 做归一化处理,现在 pw_b中保存的是词频率
        print("词对个数: " , len(self.bs))
        # for i in self.bs :
        #     print(i.get_wi() , i.get_wj())





    def update_biterm(self, bi):  # 为一个词对进行主题更新
        # print('-----------')
        # print(bi.get_wi(),bi.get_wj())
        self.reset_biterm_topic(bi)  # 将该词对bi移出所属主题，将其暂时归类为-1
        # comput p(z|b),相当于论文中计算Zb
        pz = Pvec()
        self.comput_pz_b(bi, pz)  # 计算出来的结果，直接作用在pz上。
        # print(pz) #pz是一个三个具体的数，如果主题的个数是5的话，那么pz就是5个具体的数。
        # print(pz.to_vector(), len(pz.to_vector()))  # pz.to_vector()表示将三个数转成向量。
        # sample topic for biterm b
        k = mul_sample(pz.to_vector())  # k表示根据pz算出三个数中最大的主题。。
        # print(k)
        # print('-----------')
        # print('\n')
        self.assign_biterm_topic(bi, k)  # 更新论文中的Nz,N_wiz,N_wjz，把该词对分配给上面那个主题k重新分配主题

    def reset_biterm_topic(self, bi):
        k = bi.get_z()  # 获取该词对对应的主题标签
        w1 = bi.get_wi()
        w2 = bi.get_wj()

        self.nb_z[k] -= 1  # 词对库中k主题数目-1
        self.nwz[k][w1] -= 1  # k主题下w1单词数-1
        self.nwz[k][w2] -= 1  # k主题下w2单词数-1
        # 以上减值意味将该词对移出k主题

        assert (self.nb_z[k] > -10e-7 and self.nwz[k][w1] > -10e-7 and self.nwz[k][w2] > -10e-7)
        bi.reset_z()  # 将该词对的主题号置为-1

    def assign_biterm_topic(self, bi, k):
        # bi是每一个词对，K是主题的个数。
        bi.set_z(k)  # 将该词对的主题标记为z
        w1 = bi.get_wi()  # 词对中的一个词
        w2 = bi.get_wj()  # 词对中的第二个词
        self.nb_z[k] += 1  # self.nb_z: 表示的是在词对库中，每个主题出现的次数。
        self.nwz[k][w1] += 1  # k主题下的w1单词数量+1
        self.nwz[k][w2] += 1  # k主题下的w2单词数量+1

    def comput_pz_b(self, bi, pz):
        # 计算
        pz.resize(self.K)
        w1 = bi.get_wi()  # 取到词对中的第一个词编号。
        w2 = bi.get_wj()  # 取到词对中的第二个词编号。

        for k in range(self.K):
            if (self.has_background and k == 0):
                pw1k = self.pw_b[w1]
                pw2k = self.pw_b[w2]
            else:
                pw1k = (self.nwz[k][w1] + self.beta) / (2 * self.nb_z[k] + self.W * self.beta);
                pw2k = (self.nwz[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.W * self.beta);
                # self.nwz[k][w1]:每个主题下的w1单词出现次数
                # self.nb_z[k]:词库中主题k的出现次数，即:有多少词对被分配给了主题k

            pk = (self.nb_z[k] + self.alpha) / (len(self.bs) + self.K * self.alpha);  # len(self.bs)表示的是在文档中以后多少的词对
            pz[k] = pk * pw1k * pw2k;  # 原文公式4

    def save_res(self, res_dir):
        pt = res_dir + "pz"
        print("\nwrite p(z): " + pt)
        self.save_pz(pt)

        pt2 = res_dir + "pw_z"
        print("write p(w|z): " + pt2)
        self.save_pw_z(pt2)

    # p(z) is determinated by the overall proportions of biterms in it
    # 函数计算的是每个主题的分布。
    def save_pz(self, pt):
        pz = Pvec(pvec_v=self.nb_z)
        pz.normalize(self.alpha)
        pz.write(pt)

    # 函数计算的是每个主题下各个单词的分布
    def save_pw_z(self, pt):
        pw_z = np.ones((self.K, self.W))  # 生成5行2700列的矩阵。用来保存每个主题中，各个单词出现的概率。
        wf = open(pt, 'w')
        for k in range(self.K):
            for w in range(self.W):
                pw_z[k][w] = (self.nwz[k][w] + self.beta) / (self.nb_z[k] * 2 + self.W * self.beta)  # 计算每个词在这个主题中出现的概率。
                wf.write(str(pw_z[k][w]) + ' ')
            wf.write("\n")


