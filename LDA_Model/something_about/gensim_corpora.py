from gensim import corpora
import jieba

text = ["今天天气真好真好","今天吃什么","明天天气真好真好","明天晚天气真好真好"]
text_list = []
for line in text :
    line_split = jieba.lcut(line)
    text_list.append(line_split)

print(text_list)

dictionary = corpora.Dictionary(text_list)#Dictionary传入参数：句子列表
print("len(dictionary): " , len(dictionary))
#dictionary完成建立词典 -- id:单词
for key in dictionary.keys() :
    print("key: " , key , "dictionary[key]: " , dictionary[key] , "dictionary.dfs[key]: " , dictionary.dfs[key])
    #(word_id , word , word_tf)
    #dfs[x]表示单词索引对应的单词出现的文档次数

print("dictionary.token2id: " , dictionary.token2id['今天'])#[word : id]
print("dictionary,id2token: " , dictionary.id2token)#[id : word]
print("dictionary.num_docs: " , dictionary.num_docs)#文档数量
print("dictionary.num_pos: " , dictionary.num_pos)#s所有词个数(含重)
print("dictionary.num_: " , dictionary.num_nnz)#所有文档中不重复词 次数之和(不重复是指文档中不重复)

corpus = [dictionary.doc2bow(text) for text in text_list]#文档向量化
print("corpus: " , corpus)

