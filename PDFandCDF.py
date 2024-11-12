import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import norm


df = pd.read_csv('preprocessed_marketing_campaign.csv')

data = df['Income']
mean = statistics.mean(data)
std_dev = statistics.stdev(data)

x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)

pdf = norm.pdf(x, mean, std_dev)
cdf = norm.cdf(x, mean, std_dev)

figure, axs = plt.subplots(1, 2, figsize = (10,6), layout = 'constrained')
axs[0].plot(x, pdf, label = 'PDF', color = 'blue')
axs[0].set_title('PDF of Income Distribution')
axs[0].set_ylabel('Probability Density')
axs[0].set_xlabel('Income')
axs[0].axvline(x = mean, linestyle = '--', color = 'red', label = f'Mean income: {round(mean, 2)}')
axs[0].legend()

axs[1].plot(x, cdf, label = 'CDF', color = 'blue')
axs[1].set_title('CDF of Income Distribution')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_xlabel('Income')   
axs[1].axvline(x = mean, linestyle = '--', color = 'red', label = f'Mean income: {round(mean, 2)}')
axs[1].legend()

plt.show()