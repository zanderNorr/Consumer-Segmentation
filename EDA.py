from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

df = pd.read_csv("preprocessed_marketing_campaign.csv")

# 2-D Scatter plot consists of Income (dollars) vs Amount of wine (dollars) purchased.

x = df["Income"]

print(x.shape)

categories = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
row = 0
row_max = 1
col = 0
col_max = 2
fig, axs = plt.subplots(2, 3, figsize=(10, 6), layout="tight")
for category in categories:
    if col > col_max:
        col = 0
        row += 1
    y = df[category]
    axs[row, col].scatter(x, y, color="blue", s=20, marker = 'o', alpha = 0.25)
    axs[row, col].axvline(
        x=statistics.mean(x), linestyle="--", color="red", label="Average Income"
    )
    axs[row, col].set_xlabel("Income per year ($)")
    axs[row, col].set_ylabel(f"Dollars spent on {category} ($)")
    axs[row, col].set_title(
        f"Income Per Year vs Dollars Spent On {category}", size="10"
    )
    axs[row, col].legend()
    col += 1

plt.show()

# 2-D scatterplot consisitng of income vs total purchases (feature engineering of all subgroups)
y = df["MntTotalPurchased"]
plt.xlabel("Income (Dollars)")
plt.ylabel("Total Purchases (Dollars)")
plt.title("Income (Dollars) vs Total Purchases (Dollars)")
plt.scatter(x, y, color="blue", s=20, marker = 'o', alpha = 0.25)
plt.axvline(x=statistics.mean(x), linestyle="--", color="red", label="Average Income")
plt.legend()
plt.show()

# Histogram of income vs total deals purchased

x = df['Income']
y = df['NumDealsPurchases']
total_num_deals = len(y.unique())
plt.hist(y, bins = total_num_deals)
plt.show()

valid_incomes = df['Income'] < statistics.mean(df['Income'])
df = df[valid_incomes]

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection = '3d')
x = df['Year_Birth']
y = df['MntTotalPurchased']
z = df['Income']
ax.scatter(x, y, z, c='b', marker = 'o', alpha = 0.25)
plt.show()
