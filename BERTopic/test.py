i = 1

def a(num) :
    for i in range(num) :
        i = i + 1
    return i


b = a(4)#4

def c(i) :
    c = b+i
    return c

print(c(4))