import numpy as np

arr = np.array([[1,2],
                [9,6]])
print(arr)
maxr = np.amax(arr, axis=1)
print(maxr)