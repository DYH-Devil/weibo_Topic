'''
用于保存用户user_id
'''
import pandas as pd

file = pd.read_csv('../data/crawl/data.csv' , encoding = 'utf-8')


user_id = file['user_id']
user_id.drop_duplicates(inplace = True)
user_list = user_id.to_list()

user_data = open('../data/crawl/user_id.txt' , 'w' , encoding = 'utf-8')

for user in user_list :
    user_data.write(str(user) + '\n')


