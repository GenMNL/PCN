import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(2, 5), index=np.arange(1, 3), columns=np.arange(1, 6))
print(df)
array = np.array([1, 1, 1, 1, 1])
df.loc[2] = array

print(df)
