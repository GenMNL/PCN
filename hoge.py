import pandas as pd
import numpy as np

a = [1, 1, 1, 1, 1]
a = np.array(a)
df = pd.DataFrame(np.zeros((1, 5)), columns=np.arange(1, 6))
df.loc[0] = a
print(df)
