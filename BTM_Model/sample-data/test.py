file = open('res_file.dat' , 'r' , encoding = 'utf-8')

for index , line in enumerate(file.readlines()) :
    if('李易峰' in line and '地震' in line) :
        print(index)