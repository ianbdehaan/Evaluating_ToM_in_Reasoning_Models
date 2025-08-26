import pandas as pd
from numpy import nan

df = pd.read_csv('../data/instructions.csv')[1:].reset_index()

# get the descriprion for the H1 and H3 to clone it to the others
description_H1 = df['Scenario'][27]
description_H3 = df['Scenario'][43]
df.loc[28:42,'Scenario'] = description_H1
df.loc[44:,'Scenario'] = description_H3

#combine Scenario with the first question
df['Q_1'] = (df['Scenario'] + '\n' + df['Q_1'])
#drop columns that are not useful
df = df.drop(['index','Scenario', 'Q_3', 'A_3'], axis=1)

# get the system prompts and make a column with the correct ones
S = pd.read_csv('../data/instructions.csv').iloc[0].Scenario.split(': ')
S = S[1].strip("For IM"), S[2].strip('"')
df['S'] = S[0]
df['S'] = df['S'].where(df.Type != 'H_3', S[1])
df['S'] = df['S'].where(df.Type != 'H_1', S[1])

# rename the Answer columns to Correct Answer as A will be used to store
df = df.rename(columns={'A_1': 'CA_1', 'A_2':'CA_2'})

#create empty columns to store answers
(df['A_1'], df['A_2'], df['R_1'], df['R_2']) = ('', '', '', '')
df = df[['Type', 'S', 'Q_1', 'CA_1', 'A_1', 'R_1', 'Q_2', 'CA_2', 'A_2', 'R_2', 'Level', 'Q_type', 'Deviation']]
df.to_csv('queries_treated.csv', index=False)
print(df)
