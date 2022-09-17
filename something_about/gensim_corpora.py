from gensim import corpora
import jieba

text = ["今天天气真好","今天吃什么","明天天气不好","明天晚上吃什么"]
text_list = []
for line in text :
    line_split = jieba.lcut(line)
    text_list.append(line_split)

print(text_list)

dictionary = corpora.Dictionary(text_list)#Dictionary传入参数：句子列表
print(len(dictionary))
#dictionary完成建立词典 -- id:单词
for key in dictionary.keys() :
    print(key , dictionary[key] , dictionary.dfs[key])
    #dfs[x]表示单词索引对应的单词次数