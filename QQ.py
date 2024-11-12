from scipy.stats import shapiro
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
df = pd.read_csv('preprocessed_marketing_campaign.csv')

data = df['Income']

stat, p_value = shapiro(data)

fig, axs = plt.subplots(1, 2, layout = 'constrained')

x = df['Income']

axs[0].hist(x, color ='blue', edgecolor = 'white')
axs[0].set_title('Histogram of Income Vs. Frequency')
axs[0].set_ylabel('Frequency')
axs[0].set_xlabel('Income (Dollars)')
# creating Q-Q plot to prove normality

stats.probplot(x, dist='norm', plot = plt)
axs[1].set_title('Normal Q-Q plot')
axs[1].set_xlabel('Theoretical quantiles')
axs[1].set_ylabel('Ordered Values')
axs[1].grid(True)
plt.show()

#The income of customers is normalized, with the extreme tail ends being deviated.


