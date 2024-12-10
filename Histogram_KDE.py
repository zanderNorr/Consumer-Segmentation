import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import statistics
df = pd.read_csv("preprocessed_marketing_campaign.csv")

x = df["Income"]

#histogram of income w/ varying bin sizes
bins = [10, 20, 30]
columns = [0, 1, 2]
average_income = statistics.mean(x)
std_dev = statistics.stdev(x)
print(f'Average Income: ${average_income}')
print(f'Std. Dev: ${std_dev}')
fig, axs = plt.subplots(1, 3, figsize=(10, 6), layout="constrained")
for i, j in zip(columns, bins):
    axs[i].hist(x, bins=j, edgecolor="white")
    axs[i].axvline(
        average_income,
        color="red",
        linestyle="--",
        label=f"Average Income = {round(average_income, 2)}",
    )
    axs[i].set_ylabel("Frequency")
    axs[i].set_xlabel("Income (Dollars)")
    axs[i].set_title(f"Income (Dollars) vs Frequency w/ Bin = {j}", size=10)
    axs[i].legend()
plt.show()


# #silverman rule of thumb:

median = np.median(x)
q1 = np.percentile(x, 25)
q3 = np.percentile(x, 75)

iqr = q3 - q1
print(iqr)
print(std_dev)
h = 0.9 * min(std_dev, (iqr / 1.34)) * (len(x) ** -0.2)
print(h)

x.plot(kind = 'kde', bw_method = h)
plt.show()
#Kernel Density Estimation (KDE) Graph.
bw = [0.025,0.05,0.1, 0.2]
for bandwidth in bw:
    x.plot(kind = 'kde', bw_method = bandwidth)
x.plot(kind = 'kde', bw_method = 'silverman')
plt.legend(bw, title = 'Bandwidth Value')
plt.xlim(0, max(x))
plt.ticklabel_format(style='plain', axis='y') 
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution Using Kernel Density Estimation')
plt.show()