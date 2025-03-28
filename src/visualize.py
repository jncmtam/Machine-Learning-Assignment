import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train.csv')
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title("Distribution of House Prices")
plt.savefig('output/price_distribution.png')
plt.show()
