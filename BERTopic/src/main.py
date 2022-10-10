#encoding=GBK
import os.path

from  reduce_topic_num import merge_topic
from reduce_topic_num import merge_len
import joblib
from PIL import Image
from matplotlib import pyplot as plt

docs_df , t_size , docs_per_topic , c_tfidf ,  similarity , topic_words = merge_topic(merge_len)

#-----------------------���չʾ--------------------------------
print("�������µ��ȵ��")
del topic_words[-1]#ȥ������
for topic , words in topic_words.items() :
    print(topic , words)

print("-" * 50)

print("��������ֲ�:")
topic_dist = t_size.loc[t_size['Topic'] != -1]
print(topic_dist)

print("-" * 50)

print("�������ƶȾ���:(������)")
print(similarity)
print(similarity.shape)

print("-" * 50)

print("�ϲ������Ļ���-�ĵ���")
docs_per_topic = docs_per_topic.loc[docs_per_topic['Topic'] != -1]
print(docs_per_topic)

#-----------------------------���Ʒ���---------------------------------
num_topic = len(docs_per_topic['Topic'].to_list())
for i in range(num_topic) :
    pic_name = '../res_save/wordcloud/topic'+str(i + 1)+'_cloud.png'
    if os.path.exists(pic_name) :
        img = Image.open(pic_name)
        plt.figure("Image" + str(i + 1))
        plt.imshow(img)
        plt.axis('on')
        plt.title("Topic" + str(i + 1) + "Cloud")
        plt.show()
    else :
        print("û���ҵ�ͼƬ")