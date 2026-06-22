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

    def plot_revenue_trend(self, ax):
        weekly = self.df.groupby('week')['revenue'].sum().reset_index()
        ax.plot(weekly['week'], weekly['revenue'],
            color="#F715AB", linewidth=2, marker='o', label='Revenue')
        ax.set_title("Weekly Revenue Trend")
        ax.set_xlabel("Week")
        ax.set_ylabel("Revenue")
        ax.legend()

    def plot_product_breakdown(self, ax):
        product_rev = self.df.groupby('product')['revenue'].sum().sort_values()
        ax.barh(product_rev.index, product_rev.values,
                color=['#070F34', '#0313A6', '#9201CB', '#F715AB', '#34EDF3'])
        ax.set_title("Revenue By Product")
        ax.set_xlabel("Revenue ($)")

    def plot_regional_comparison(self, ax):
        pivot = pd.pivot_table(
        self.df,
        values= 'revenue',
        columns= 'region',
        index= 'product',
        aggfunc='sum'
        )
        pivot.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_title("Revenue by Region & Product")
        ax.set_xlabel("Region")
        ax.set_ylabel("Revenue ($)")
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=0)

    def plot_unit_distribution(self, ax):
        sns.histplot(data=self.df, x='units', bins=14, ax=ax, color='#F715AB')
        ax.set_title("Units per Transaction")
        ax.set_xlabel("Units")
        

    def plot_revenue_heatmap(self, ax):
        # heatmap: products (rows) × months (cols) = total revenue
        # hint: pd.pivot_table then sns.heatmap
        pivot = pd.pivot_table(
            self.df,
            values='revenue',
            columns= 'month',
            index='product',
            aggfunc='sum'
        )

        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='coolwarm', ax=ax)
        ax.set_title("Revenue: Product × Month")

    def plot_top_days(self, ax):
        daily_rev = (self.df.groupby('date')['revenue']
                     .sum()
                     .sort_values(ascending=False)
                     .head(10))
        labels = [d.strftime('%b %d') for d in daily_rev.index]
        ax.bar(labels, daily_rev.values, color='#0313A6')
        ax.set_title("Top 10 Revenue Days")
        ax.set_ylabel("Revenue ($)")
        ax.tick_params(axis='x', rotation=45)

    def render(self, filename: str = 'dashboard.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sales Dashboard 2026', fontsize=16, fontweight='bold')

        self.plot_revenue_trend(axes[0, 0])
        self.plot_product_breakdown(axes[0, 1])
        self.plot_regional_comparison(axes[0, 2])
        self.plot_unit_distribution(axes[1, 0])
        self.plot_revenue_heatmap(axes[1, 1])
        self.plot_top_days(axes[1, 2])

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Dashboard saved to {filename}")


dashboard = SalesDashboard(df)
dashboard.render('dashboard.png')