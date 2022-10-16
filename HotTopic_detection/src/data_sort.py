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
    date_change = datetime.datetime.strptime(date , "%d/%m/%Y %H:%M:%S")
    date_change = str(date_change)
    return date_change


#对微博进行分类(热点话题 和 非热点话题)
def WEIBO_CLASSIFY(data_df , alpha = 0.5 , beta = 0.5 , theta = 0.5) :
    '''
    :param data_df: 微博文档集:DataFrame
    :param alpha: 微博转发数权重
    :param beta: 微博评论数权重
    :param theta: 微博点赞数权重
    :return: None 分类完的微博数据集直接保存
    '''

    #先对微博数据集的日期进行排序
    data_df = data_df[['date', 'text', 'repostsnum', 'commentsnum', 'attitudesnum']]
    data_df['date'] = data_df['date'].map(change)
    data_df = data_df.sort_values(by='date')

    # 将三个属性为空的值填充0
    data_df['repostsnum'].fillna(0 , inplace = True)
    data_df['commentsnum'].fillna(0 , inplace = True)
    data_df['attitudesnum'].fillna(0 , inplace = True)

    #计算spread热度值
    data_df['spread'] = alpha * np.log(data_df['repostsnum']) + \
                     beta * np.log(data_df['commentsnum']) + \
                     theta * np.log(data_df['attitudesnum'])

    data_df.loc[data_df['spread'] == -np.inf] = 0

    hot_topic = data_df.loc[(data_df['spread'] > 0)]

    hot_topic.to_csv('../data/hotopic.csv' , index = False)#保存热门话题数据

