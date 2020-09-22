import numpy as np 

a = np.arange(16).reshape((4, 4))
print(a, 'full array')
print(a[::2], 'even rows')
print(a[1::2], 'odd rows')
print(a[:, 0::2], 'even columns')
print(a[:, 1::2], 'odd columns')
print(a[::2, 0::2], ' even rows & even columns')

print('done')