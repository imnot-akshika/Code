import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# generate sample data
np.random.seed(42)
n = 200

products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
regions  = ['North', 'South', 'East', 'West']
reps     = [f'Rep_{i:02d}' for i in range(10)]

dates = [
    datetime(2026, 1, 1) + timedelta(days=int(i))
    for i in np.random.randint(0, 180, n)
]

df = pd.DataFrame({
    'date':     dates,
    'product':  np.random.choice(products, n),
    'region':   np.random.choice(regions, n),
    'rep':      np.random.choice(reps, n),
    'units':    np.random.randint(1, 20, n),
    'price':    np.random.choice([999.99, 29.99, 79.99, 399.99, 149.99], n),
})

df['revenue'] = df['units'] * df['price']

def top_products(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(n).reset_index()

def regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['revenue'] = df['units'] * df['price']
    regional_summary = df.groupby('region').agg(
        total_revenue = ('revenue', 'sum'),
        total_units = ('units', 'sum'),
        num_transactions = ('units', 'count')
    ).reset_index()

    regional_summary['avg_order_value'] = (
        regional_summary['total_revenue'] / regional_summary['num_transactions']
    )

    return regional_summary

def rep_performance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() 

    # for each rep: total_revenue, num_sales, avg_revenue_per_sale
    # add a 'tier' column: 'Gold' if total_revenue > 50000, else 'Silver'
    df['revenue'] = df['units'] * df['price']
    rep_performance = df.groupby('rep').agg(
        total_revenue = ('revenue', 'sum'),
        num_sales = ('units', 'count')
    ).reset_index()

    rep_performance['avg_revenue_per_sale'] = (
        rep_performance['total_revenue'] / rep_performance['num_sales']
    )

    rep_performance['tier'] = np.where(
        rep_performance['total_revenue'] > 50000,
        'Gold',
        'Silver'
    )

    return rep_performance

def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # group by month, return total revenue per month
    # hint: df['date'].dt.month
    result = df.groupby(df['date'].dt.month)['revenue'].sum().reset_index()
    result.columns = ['month', 'total_revenue']
    return result

def product_region_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # pivot table: products as rows, regions as columns, revenue as values
    # hint: pd.pivot_table()
    product_region_matrix = pd.pivot_table(
        df,
        values= 'revenue',
        columns= 'region',
        index= 'product'
    )

    return product_region_matrix


def find_best_day(df: pd.DataFrame) -> tuple[str, float]:
    df = df.copy()
    # find the date with highest total revenue
    # return (date_string, total_revenue)
    daily_revenue = df.groupby(df['date'].dt.date)['revenue'].sum()

    best_day = daily_revenue.idxmax()
    best_revenue = daily_revenue.max()

    return str(best_day), float(best_revenue)



#Example Usage
print("Top products:")
print(top_products(df))

print("\nRegional summary:")
print(regional_summary(df))

print("\nRep performance:")
print(rep_performance(df).sort_values('total_revenue', ascending=False))

print("\nMonthly trend:")
print(monthly_trend(df))

print("\nProduct × Region matrix:")
print(product_region_matrix(df).round(0))

date, rev = find_best_day(df)
print(f"\nBest day: {date} — ${rev:,.2f}")