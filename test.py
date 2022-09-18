import jieba

texts = "李易峰嫖娼被抓"
res = jieba.cut(texts , cut_all = True)
for i in res :
    print(i)