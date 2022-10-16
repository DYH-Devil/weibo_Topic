#encoding=GBK
import pandas as pd
import numpy as np


file = pd.read_csv('../data/weibo_content.csv' , encoding = 'utf-8')

alpha = beta = theta = 0.5

file['repostsnum'].fillna(0 , inplace = True)
file['commentsnum'].fillna(0 , inplace = True)
file['attitudesnum'].fillna(0 , inplace = True)


file['spread'] = alpha * np.log(file['repostsnum']) + \
                 beta * np.log(file['commentsnum']) + \
                 theta * np.log(file['attitudesnum'])

file.loc[file['spread'] == -np.inf] = 0

weibo_content = file.loc[(file['repostsnum'].values > 500) &
                         (file['commentsnum'].values > 500) &
                         (file['attitudesnum'].values > 500) &
                         (file['spread'] > 0)]

print(weibo_content)

