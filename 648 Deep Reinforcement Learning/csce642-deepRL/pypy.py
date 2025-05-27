import numpy as np
A = np.arange(10)
A[9] = 8
print(A)
print(np.argmax(A))