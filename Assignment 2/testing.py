import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('VeerArda_boarddiagram3.tsv', sep = "\t")
#df = df.T
print(df.shape)

pos_col_num = 7
sample = df.iloc[:, pos_col_num]
fs = 500

x = [i/fs for i in range(len(sample))]

#plotting a single sample
plt.plot(x, sample)
plt.title('heartrate sample')
plt.xlabel('Seconds')
plt.ylabel('Amplitude')
plt.show()






