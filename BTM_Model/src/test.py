'''
@Author: 一蓑烟雨任平生
@Date: 2020-02-18 17:08:33
@LastEditTime: 2020-03-08 15:54:21
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /BTMpy/src/test.py
'''
# -*- coding: utf-8 -*-
import time
from Model import *
import sys
import indexDocs
import topicDisplay
import os

def usage() :
    print("Training Usage: \
    btm est <K> <W> <alpha> <beta> <n_iter> <save_step> <docs_pt> <model_dir>\n\
    \tK  int, number of topics, like 20\n \
    \tW  int, size of vocabulary\n \
    \talpha   double, Pymmetric Dirichlet prior of P(z), like 1.0\n \
    \tbeta    double, Pymmetric Dirichlet prior of P(w|z), like 0.01\n \
    \tn_iter  int, number of iterations of Gibbs sampling\n \
    \tsave_step   int, steps to save the results\n \
    \tdocs_pt     string, path of training docs\n \
    \tmodel_dir   string, output directory")


def BTM(argvs):
    if(len(argvs)<4):
        usage()
    else:
        if (argvs[0] == "est"):
            K = argvs[1]
            W = argvs[2]
            alpha = argvs[3]
            beta = argvs[4]
            n_iter = argvs[5]
            save_step = argvs[6]
            docs_pt = argvs[7]
            dir = argvs[8]
            #传入参数完毕

            print("===== Run BTM, K="+str(K)+", W="+str(W)+", alpha="+str(alpha)+", beta="+str(beta)+", n_iter="+str(n_iter)+", save_step="+str(save_step)+"=====")
            clock_start = time.perf_counter()

            model = Model(K, W, alpha, beta, n_iter, save_step)#初始化模型
            model.run(docs_pt,dir)
            clock_end = time.perf_counter()
            print("procedure time : "+str(clock_end-clock_start))
        else:
            usage()

if __name__ ==  "__main__":
    mode = "est"
    K = 4#主题数数量
    W = None
    alpha = K / 50#alpha =  K / 50
    beta = 0.01
    n_iter = 1000 # beta:0.1,iter:30次效果较理想
    save_step = 100
    dir = "../output/"
    input_dir = "../sample-data/"
    model_dir = dir + "GSDMM/" #模型存放的文件夹
    voca_pt = dir + "voca.txt" #生成的词典
    dwid_pt = dir + "doc_wids.txt" #每篇文档由对应的序号单词组成
    doc_pt = input_dir + "res_file.dat" #输入的文档



    print("=============== Index Docs =============")
    # W生成的词典
    W = indexDocs.run_indexDocs(['indexDocs',doc_pt,dwid_pt,voca_pt]) #返回的是词典w2id长度
    #doc_pt:输入文档,已分词的文档
    #dwid_pt:将输入文档转化成数字序列的文档
    #voca_pt:id : word 词典
    #返回的是字典长度


    print("W : "+str(W))

    argvs = []
    argvs.append(mode)
    argvs.append(K)
    argvs.append(W)
    argvs.append(alpha)
    argvs.append(beta)
    argvs.append(n_iter)
    argvs.append(save_step)
    argvs.append(dwid_pt)
    argvs.append(model_dir)


    print("=============== Topic Learning =============")
    BTM(argvs)

    print("================ Topic Display =============")
    topicDisplay.run_topicDicplay(['topicDisplay',model_dir,K,voca_pt])