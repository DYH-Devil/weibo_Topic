a = 1
def b(num) :
    global a
    a += num

def c() :
    global a
    a = 5
    for i in range(1) :
        b(2)
        print(a)
    print(a)

c()