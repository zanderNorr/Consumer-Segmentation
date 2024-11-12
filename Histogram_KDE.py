import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import statistics
df = pd.read_csv("preprocessed_marketing_campaign.csv")

x = df["Income"]

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

#Kernel Density Estimation (KDE) Graph.
bw = [0.025,0.05,0.1, 0.2]
for bandwidth in bw:
    x.plot(kind = 'kde', bw_method = bandwidth)
plt.legend(bw, title = 'Bandwidth Value')
plt.xlim(0, max(x))
plt.ticklabel_format(style='plain', axis='y') 
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution Using Kernel Density Estimation')
plt.show()