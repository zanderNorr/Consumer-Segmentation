import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("preprocessed_marketing_campaign.csv")

data = df['Income']

x = np.array(df["Income"])

scaler = StandardScaler()

x_normalized = scaler.fit_transform(x.reshape(-1,1)).flatten()

#now data normalized

median = np.median(x_normalized)
q1 = np.percentile(x_normalized, 25)
q3 = np.percentile(x_normalized, 75)
IQR = q3 - q1
std_dev = np.std(x_normalized)
h = 0.9 * min(std_dev, (IQR / 1.34)) * (len(x) ** -0.2)
print(h)
data.plot(kind = 'kde', bw_method = h, label = f'KDE (Optimized Silverman) = {round(h, 3)}', color = 'red')
data.plot(kind = 'hist', density = True, bins = 30, label = 'Histogram', color='grey')
# plt.legend(h, title = "Silvermans' Bandwidth Value")
plt.xlim(0, max(x))
plt.ticklabel_format(style='plain', axis='y') 
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title("Histogram KDE with Silvermans' Bandwidth Value of Consumer Income")
plt.legend(loc = 'best')
plt.show()