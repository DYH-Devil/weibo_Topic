#encoding=GBK
import pandas as pd
import datetime

def change(date) :
    date_change = datetime.datetime.strptime(date , "%Y/%m/%d %H:%M")
    date_change = str(date_change)
    return date_change

data = pd.read_csv('../data/data.csv' , encoding = 'gbk')
print(data.head())

data_df = data[['����ʱ��' , '΢������']]
data_df['����ʱ��'] = data_df['����ʱ��'].map(change)
data_df = data_df.sort_values(by = "����ʱ��")

print(data_df)

start , end = pd.to_datetime(["2022/09/10 00:00" , "2022/09/15 00:00"] ,
                             format = "%Y/%m/%d %H:%M")
print(type(start))
print(start)
print(type(end))
print(end)

print(type(data_df['����ʱ��']))

# dataset = data_df[
#     (data_df['����ʱ��'] >= start &
#      data_df["����ʱ��"] <= end)
# ]
dataset = data_df[
    (pd.to_datetime(data_df['����ʱ��']) >= start) &
    (pd.to_datetime(data_df['����ʱ��']) <= end)
]

print(dataset)

print(len(dataset))


