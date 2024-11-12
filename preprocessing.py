import pandas as pd
import matplotlib.pyplot as plt
#read CSV with the column seperator being '\t'
df = pd.read_csv('marketing_campaign.csv', delimiter='\t')

#strips columns of any space
df.columns = df.columns.str.strip()

#strips row entries of any whitespace
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

#only birth years after 1900 are valid
valid_birth_year = df['Year_Birth'] > 1900
df = df[valid_birth_year]

#set income to an integer and only income < $200,000 is valid
df = df[df['Income'] != '']
df['Income'] = df['Income'].astype(int)
valid_income = df['Income'] <= 165000 
df = df[valid_income]

#2n Cycle is classifed as Masters, so its replaced
mask = df['Education'].str.contains('2n Cycle')
df.loc[mask, 'Education'] = 'Master'

#Basic degree is high school diploma, so its replaced
mask = df['Education'].str.contains('Basic')
df.loc[mask, 'Education'] = 'High School Diploma'

#Graduation is replaced with Undergraduate Degree
mask = df['Education'].str.contains('Graduation')
df.loc[mask, 'Education'] = 'Undergraduate Degree'

#preprocess the 'Marital_Status' column with only 'Single', 'Married', 'Divorced', 'Widow'
#categories such as 'Alone' 'Absurd' 'YOLO' will be classifed as 'Single'
#categories such as 'Together' will be classifed as 'Single' as well because no proof of marrage.
mask = df['Marital_Status'].str.contains('Alone|Absurd|YOLO|Together', regex=True)
df.loc[mask, 'Marital_Status'] = 'Single'


total_amt_purchased = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['MntTotalPurchased'] = total_amt_purchased

total_dependents = df['Kidhome'] + df['Teenhome']
df['TotalDependents'] = total_dependents

df.to_csv('preprocessed_marketing_campaign.csv', index=False)