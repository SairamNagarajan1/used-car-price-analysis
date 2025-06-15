# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Step 4: Load the Dataset
# Read dataset and drop the first unnamed index column
df = pd.read_csv('output.csv')
df = df.iloc[:, 1:]
df.head()

# Step 5: Assign Column Headers
headers = ["symboling", "normalized-losses", "make", 
           "fuel-type", "aspiration", "num-of-doors",
           "body-style", "drive-wheels", "engine-location",
           "wheel-base", "length", "width", "height", "curb-weight",
           "engine-type", "num-of-cylinders", "engine-size", 
           "fuel-system", "bore", "stroke", "compression-ratio",
           "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers

# Step 6: Check for Missing Values
print(df.isnull().sum())  # Check missing values
print(df.isna().sum())

# Step 7: Convert MPG to L/100km
df['city-mpg'] = 235 / df['city-mpg']
df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)

# Step 8: Convert Price Column to Integer
# Remove rows with missing price values
df = df[df['price'] != '?']
df['price'] = df['price'].astype(int)

# Step 9: Normalize Features
# Normalize selected columns to bring them to the same scale
df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / df['height'].max()

# Binning price into categories
bins = np.linspace(min(df['price']), max(df['price']), 4)
group_names = ['Low', 'Medium', 'High']
df['price-binned'] = pd.cut(df['price'], bins, labels=group_names, include_lowest=True)

# Plot histogram of binned prices
plt.figure(figsize=(8,4))
plt.hist(df['price-binned'], bins=3, edgecolor='black')
plt.title("Price Binned Distribution")
plt.xlabel("Price Category")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# Step 10: Convert Categorical Data to Numerical
# One-hot encode categorical features (example: fuel-type)
fuel_dummies = pd.get_dummies(df['fuel-type'], prefix='fuel')
df = pd.concat([df, fuel_dummies], axis=1)

# Step 11: Data Visualization
# Boxplot of price by drive wheels
plt.figure(figsize=(8,4))
sns.boxplot(x='drive-wheels', y='price', data=df)
plt.title("Boxplot: Drive Wheels vs Price")
plt.show()

# Scatter plot of engine-size vs price
plt.figure(figsize=(8,4))
plt.scatter(df['engine-size'], df['price'], alpha=0.5)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Step 12: Grouping Data by Drive-Wheels and Body-Style
grouped_data = df[['drive-wheels', 'body-style', 'price']].groupby(
    ['drive-wheels', 'body-style'], as_index=False).mean()
print(grouped_data)

# Step 13: Create a Pivot Table & Heatmap
pivot = grouped_data.pivot(index='drive-wheels', columns='body-style', values='price')
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdBu", linewidths=.5)
plt.title("Heatmap: Average Price by Drive-Wheel and Body-Style")
plt.show()

# Step 14: Perform ANOVA Test
# Compare price means of Honda and Subaru cars
anova_df = df[['make', 'price']]
grouped_anova = anova_df.groupby('make')
anova_result = stats.f_oneway(grouped_anova.get_group('honda')['price'],
                              grouped_anova.get_group('subaru')['price'])
print("ANOVA test result:", anova_result)

# Regression plot of engine-size vs price
plt.figure(figsize=(8,4))
sns.regplot(x='engine-size', y='price', data=df)
plt.title("Regression Plot: Engine Size vs Price")
plt.ylim(0, )
plt.grid(True)
plt.show()
