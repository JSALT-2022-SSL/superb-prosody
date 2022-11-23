import pandas as pd

df = pd.read_csv("CMU_MOSEI_Labels.csv", index_col=0)

i  = df[df.label2b == 0].index
df = df.drop(i)

df = df.drop('label2a', axis=1)
df = df.drop('label6', axis=1)
df = df.drop('label7', axis=1)

df = df.replace({'label2b': -1}, 0)
df = df.rename(columns={"label2b":"label2a"})

df.to_csv("CMU_MOSEI_Labels_new.csv")
