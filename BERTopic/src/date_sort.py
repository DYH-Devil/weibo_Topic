#encoding=GBK
import pandas as pd
import datetime

def change(date) :
    date_change = datetime.datetime.strptime(date , "%d/%m/%Y %H:%M:%S")
    date_change = str(date_change)
    return date_change

data = pd.read_csv('../data/weibo_data.csv' , encoding = 'utf-8')

data_df = data[['date' , 'text']]
data_df['date'] = data_df['date'].map(change)
data_df = data_df.sort_values(by = "date")

print(data_df)

data_df.to_csv('../data/data_sort.csv')

# start , end = pd.to_datetime(["2014/05/02 00:00" , "2014/05/03 00:00"] ,
#                              format = "%Y/%m/%d %H:%M")
#
# dataset = data_df[
#     (pd.to_datetime(data_df['date']) >= start) &
#     (pd.to_datetime(data_df['date']) <= end)
# ]
#
# print(dataset)
#
# print(len(dataset))


