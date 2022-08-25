import pandas as pd
import numpy as np

ary = np.random.randn(3, 3)

max = ary.max(axis=0)
min = ary.min(axis=0)
x_max, y_max, z_max = max[0], max[1], max[2]
x_min, y_min, z_min = min[0], min[1], min[2]
x_min, y_min = min[0], min[1]
z_max = max[2]

print(ary)
ary[:,0] -= x_min
ary[:,0] /= (x_max - x_min)
print(ary)
