#encoding=GBK
import pandas as pd
import datetime

def change(date) :
    date_change = datetime.datetime.strptime(date , "%Y/%m/%d %H:%M")
    date_change = str(date_change)
    return date_change

data = pd.read_csv('../data/data.csv' , encoding = 'gbk')
print(data.head())

data_df = data[['发布时间' , '微博正文']]
data_df['发布时间'] = data_df['发布时间'].map(change)
data_df = data_df.sort_values(by = "发布时间")

print(data_df)

start , end = pd.to_datetime(["2022/09/10 00:00" , "2022/09/15 00:00"] ,
                             format = "%Y/%m/%d %H:%M")
print(type(start))
print(start)
print(type(end))
print(end)

print(type(data_df['发布时间']))

# dataset = data_df[
#     (data_df['发布时间'] >= start &
#      data_df["发布时间"] <= end)
# ]
dataset = data_df[
    (pd.to_datetime(data_df['发布时间']) >= start) &
    (pd.to_datetime(data_df['发布时间']) <= end)
]

print(dataset)

print(len(dataset))


