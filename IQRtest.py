import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import plotly.express as px
df = pd.read_csv("marketing_campaign.csv", delimiter = '\t')

#strips columns of any space
df.columns = df.columns.str.strip()

#strips row entries of any whitespace
df = df.map(lambda x: x.strip() if isinstance(x, str) else x) 

df = df[df['Income'] != '']
df['Income'] = df['Income'].astype(int)

data_unsorted = df['Income']
data = np.array(df['Income'].sort_values())

median = np.median(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

iqr = q3 - q1
quartile_dev = (q3 - q1) / 2

print(f'Median: {median}')
print(f"Interquartile Range (IQR): {iqr}")
print(f"Quartile Deviation: {quartile_dev}")
fig = px.box(x=data_unsorted, title = 'Boxplot of Consumer Income')
fig.show()
fig = px.violin(x=data_unsorted, title = 'Violinplot of Consumer Income')
fig.show()
data = np.array(df['Year_Birth'].sort_values())
data_unsorted = df['Year_Birth']
median = np.median(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

iqr = q3 - q1
quartile_dev = (q3 - q1) / 2

print(f'Median: {median}')
print(f"Interquartile Range (IQR): {iqr}")
print(f"Quartile Deviation: {quartile_dev}")
fig = px.box(x=data_unsorted, title = 'Boxplot of Consumer Birth Year')
fig.show()
fig = px.violin(x=data_unsorted, title = 'Violinplot of Consumer Birth Year')
fig.show()