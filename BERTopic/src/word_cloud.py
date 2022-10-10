#encoding=UTF-8
'''
构建词云分析每个主题下的热点
'''

from wordcloud import WordCloud
from main import docs_per_topic

word_cloud = WordCloud(
    background_color = 'black' ,
    max_words = 10000 ,
    width = 800 ,
    height = 800 ,
    max_font_size = 200 ,
    random_state = 2022
)

doc_list = docs_per_topic['Doc'].tolist()
# print(doc)
for i , doc in enumerate(doc_list) :
    topic_cloud = word_cloud.generate(doc)
    topic_cloud.to_file('../res_save/wordcloud/topic'+str(i + 1)+'_cloud.png')