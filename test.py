import jieba
from hanlp_restful import HanLPClient

word_list = [1,2]
list = [1,2,3,1,1,2,2,3,1,2]
i_list = []
for word in word_list :
    i = 0
    if(word in list) :
        i = i+1
        i_list.append(i)

print(i_list)
