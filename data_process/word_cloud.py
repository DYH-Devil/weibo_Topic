from wordcloud import WordCloud

def WordCloud_process(text_list) :
    '''
    :param text_list: 分词后的句子列表:[[w1,w2,...] , [w1,w2,....],...]
    '''
    word_list = []
    for sentence in text_list :
        for word in sentence :
            word_list.append(word)

    word_text = " ".join(word_list)#将词库列表以空格连接，形成字符串文本

    #建立词云
    word_cloud = WordCloud(background_color = 'black' ,
                           max_words = 2000 , #最大词数
                           width = 800 , 
                           height = 800 , 
                           max_font_size = 200 , #字号大小
                           random_state = 30) #颜色随机

    word_picture = word_cloud.generate(word_text)
    return word_picture