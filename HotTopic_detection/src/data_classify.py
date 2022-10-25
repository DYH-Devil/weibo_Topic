#encoding=GBK
'''
对文件按时间排序
'''

import pandas as pd
import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def change(date) :
    date_change = datetime.datetime.strptime(date , "%Y/%m/%d %H:%M")
    date_change = str(date_change)
    return date_change


#对微博进行分类(热点话题 和 非热点话题)
def WEIBO_CLASSIFY(data_df , alpha = 0.7 , beta = 0.2 , theta = 0.1) :
    '''
    :param data_df: 微博文档集:DataFrame
    :param alpha: 微博转发数权重
    :param beta: 微博评论数权重
    :param theta: 微博点赞数权重
    :return: None 分类完的微博数据集直接保存
    '''
    data_df = data_df[['bid', 'user_id', '用户昵称', '微博正文', '转发数', '评论数', '点赞数', '发布时间']]
    #去重
    data_df.drop_duplicates(inplace = True)

    #先对微博数据集的日期进行排序
    data_df['发布时间'] = data_df['发布时间'].map(change)
    data_df = data_df.loc[(data_df['发布时间'] >= '2022-10-17 00:00:00') & (data_df['发布时间'] <= '2022-10-19 23:59:59')]
    data_df = data_df.sort_values(by='发布时间')

    # 将三个属性为空的值填充0
    data_df['转发数'].fillna(0 , inplace = True)
    data_df['评论数'].fillna(0 , inplace = True)
    data_df['点赞数'].fillna(0 , inplace = True)

    #微博spread热度值计算公式
    data_df['传播值'] = alpha * np.log(data_df['转发数']) + \
                     beta * np.log(data_df['评论数']) + \
                     theta * np.log(data_df['点赞数'])

    data_df.loc[data_df['传播值'] == -np.inf] = 0

    hot_topic = data_df.loc[(data_df['传播值'] > 0)]



    hot_topic.to_csv('../data/hotopic.csv' , index = False)#保存热门话题数据

