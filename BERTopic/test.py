import numpy as np

number = np.array([[3,4,1,6,2,9] ,
                  [5,2,6,1,7,3]])

index = number.argsort()[: ,-3 :]
print(index)
print(index[::-1])
