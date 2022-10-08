import time


filename = '../BTM_Model/sample-data/res_file.dat'
text_list = []
texts = []
row = 1000
start = 0

def read_Data(file_name) :
    file = open(file_name , 'r' , encoding = 'utf-8')
    for line in file.readlines() :
        text_list.append(line.strip())
    return text_list

def loop_read(func , sec) :
    while True :
        try :
            func(text_list)
            time.sleep(sec)
        except :
            print("执行完毕")
            break


def read_list(text_list) :
    global start
    for i in range(start , start + row) :
        try :
            texts.append(text_list[i])
        except :
            print("已读取所有数据")
            break
    start += row
    print("已读取1000行")
    return texts


text_list = read_Data(filename)

loop_read(read_list , 2)

texts = read_list(text_list)

for index , text in enumerate(texts) :
    print(index , text)