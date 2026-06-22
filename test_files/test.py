import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# generate data
np.random.seed(42)
n = 500

products  = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
regions   = ['North', 'South', 'East', 'West']
prices    = {'Laptop': 999.99, 'Mouse': 29.99, 
             'Keyboard': 79.99, 'Monitor': 399.99, 'Headphones': 149.99}

dates = [datetime(2026, 1, 1) + timedelta(days=int(i))
         for i in np.random.randint(0, 180, n)]

df = pd.DataFrame({
    'date':    dates,
    'product': np.random.choice(products, n),
    'region':  np.random.choice(regions, n),
    'units':   np.random.randint(1, 15, n),
})
df['price']   = df['product'].map(prices)
df['revenue'] = df['units'] * df['price']
df['month']   = df['date'].dt.month
df['week']    = df['date'].dt.isocalendar().week.astype(int)

class SalesDashboard:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        plt.style.use('dark_background')

