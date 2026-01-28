import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_json('features.jsonl', lines=True)
scores = df['score']

# plot the distribution of scores
sns.histplot(scores, bins=20)
plt.savefig('scores.png')

df = pd.read_json('../../../../../workspace/med/data/docs/sentences.jsonl', lines=True)
print(len(df))