import pandas as pd

res_file = open('./data/weibo_data.txt' , 'w' , encoding = 'utf-8') 

file = open('./data/weibo.txt' , 'r' , encoding='utf-8')
for line in file.readlines() :
    content = line.split('\t')
    print(content[18])
    res_file.write(content[18]+'\n')


