# 🚗 Used Car Price Analysis

A complete exploratory data analysis (EDA) and preprocessing project based on the **Imports-85** automobile dataset. This project aims to uncover insights about the factors affecting car prices and prepare the data for future machine learning models.

---

## 📂 Dataset

This project uses the [Imports-85 dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data).

It contains detailed attributes of various cars, such as:

- Make and model information  
- Technical features like engine size, body style, horsepower  
- Fuel economy, curb weight, and more  
- Final price of the car (target variable)

The raw `.data` file has been processed and saved as `output.csv` for easier use in Pandas.

---

## 🎯 Project Objective

The goal is to:

- Perform data cleaning and handle missing values
- Convert and normalize units (e.g., MPG to L/100km)
- Visualize price trends with respect to different features
- Use statistical tests like ANOVA for brand-based price differences
- Prepare the dataset for machine learning model development


---

## 🛠 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SairamNagarajan1/used-car-price-analysis.git
   cd used-car-price-analysis
pip install -r requirements.txt
