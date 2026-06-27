import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

np.random.seed(42)
n = 1000

# simulate a student performance dataset
df = pd.DataFrame({
    'student_id': range(1001, 1001 + n),
    'age':        np.random.randint(16, 25, n),
    'gender':     np.random.choice(['M', 'F', 'Other'], n, p=[0.48, 0.48, 0.04]),
    'school':     np.random.choice(['School_A', 'School_B', 'School_C'], n),
    'study_hours': np.random.normal(5, 2, n).clip(0, 12).round(1),
    'attendance': np.random.normal(80, 15, n).clip(0, 100).round(1),
    'math':       np.random.normal(65, 18, n).clip(0, 100).round(1),
    'science':    np.random.normal(62, 20, n).clip(0, 100).round(1),
    'english':    np.random.normal(68, 15, n).clip(0, 100).round(1),
    'history':    np.random.normal(64, 17, n).clip(0, 100).round(1),
    'part_time_job': np.random.choice([True, False], n, p=[0.3, 0.7]),
    'internet_access': np.random.choice([True, False], n, p=[0.85, 0.15]),
})

# add some realistic correlations
df['math'] = (df['math'] + df['study_hours'] * 2).clip(0, 100).round(1)
df['science'] = (df['science'] + df['study_hours'] * 1.5).clip(0, 100).round(1)

# introduce missing values
df.loc[np.random.choice(n, 50, replace=False), 'study_hours'] = np.nan
df.loc[np.random.choice(n, 30, replace=False), 'attendance'] = np.nan

# compute overall average
df['avg_score'] = df[['math', 'science', 'english', 'history']].mean(axis=1).round(2)
df['grade'] = pd.cut(df['avg_score'],
                     bins=[0, 50, 60, 70, 80, 100],
                     labels=['F', 'D', 'C', 'B', 'A'])


class ExplorationReport:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.clean_df = None

    def data_quality_report(self) -> dict:
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].to_dict()
        total_cells = self.df.shape[0] * self.df.shape[1]
        quality_report = ({
            'shape': self.df.shape,
            'missing_by_column': missing,
            'missing_pct': (self.df.isnull().sum().sum() / total_cells) * 100,
            'duplicates': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.astype(str).to_dict()
        })

        return quality_report

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()

        df['study_hours'] = df['study_hours'].fillna(df['study_hours'].median())
        df['attendance'] = df['attendance'].fillna(df['attendance'].mean())
        df = df.dropna()
        df = df.reset_index(drop=True)
        self.clean_df = df
        return self.clean_df

    def summary_stats(self) -> pd.DataFrame:
        if self.clean_df is None:
            self.clean()

        return self.clean_df.describe().round(2)

    def correlation_analysis(self) -> pd.DataFrame:
        if self.clean_df is None:
            self.clean()
        cols = ['study_hours', 'attendance', 'math', 'science',
                'english', 'history', 'avg_score']
        return self.clean_df[cols].corr().round(2)

    def group_analysis(self) -> dict:
        if self.clean_df is None:
            self.clean()
        df = self.clean_df
        return{
            'by_school': df.groupby('school')[['avg_score','study_hours','attendance']].mean().round(2),
            'by_gender': df.groupby('gender')['avg_score'].mean().round(2),
            'by_grade': df.groupby('grade', observed=True).agg(
                count=('student_id', 'count'),
                avg_study_hours=('study_hours','mean')
            ).round(2)
        }

    def top_and_bottom(self, n: int = 5) -> dict:
        if self.clean_df is None:
           self.clean()
        df = self.clean_df
        cols = ['student_id', 'avg_score', 'grade']
        return { 
            'top_students':    df.nlargest(n, 'avg_score')[cols],
            'bottom_students': df.nsmallest(n, 'avg_score')[cols],
            'most_studied':    df.nlargest(n, 'study_hours')[['student_id','study_hours']],
            'best_attendance': df.nlargest(n, 'attendance')[['student_id','attendance']]
        }

    def plot_report(self, filename: str = 'exploration.png'):
        if self.clean_df is None:
            self.clean()
        df = self.clean_df
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Performance', fontsize=16, fontweight='bold')

        sns.histplot(data=df, x='avg_score', hue='grade', ax=axes[0, 0], bins=20,)
        axes[0, 0].set_title("Distribution of Average Score")

        sns.regplot(data=df, x='study_hours', y='avg_score', ax=axes[0, 1], scatter_kws={'alpha': 0.5})
        axes[0, 1].set_title("Study hours vs Average Score")

        sns.boxplot(data=df, x='school', y='avg_score', ax=axes[0, 2])
        axes[0, 2].set_title("Average score by School")

        corr = df[['study_hours','attendance','math','science',
               'english','history','avg_score']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='rocket', ax=axes[1, 0])
        axes[1, 0].set_title("Correlation Heatmap")

        sns.countplot(data=df, x='grade', ax=axes[1, 1])
        axes[1, 1].set_title("Grade Distribution")

        sns.scatterplot(data=df, x='attendance', y='avg_score', hue='part_time_job', style='part_time_job', palette='mako', s=100, ax=axes[1, 2], alpha=0.5)
        axes[1, 2].set_title("Attendance vs Average score")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")
        

#Example Uasge

report = ExplorationReport(df)

print("=== Data Quality ===")
quality = report.data_quality_report()
print(f"Shape: {quality['shape']}")
print(f"Missing: {quality['missing_by_column']}")
print(f"Missing %: {quality['missing_pct']:.2f}%")
print(f"Duplicates: {quality['duplicates']}")

print("\n=== Cleaning ===")
clean = report.clean()
print(f"Clean shape: {clean.shape}")
print(f"Missing after clean: {clean.isnull().sum().sum()}")

print("\n=== Summary Stats ===")
print(report.summary_stats())

print("\n=== Correlations ===")
print(report.correlation_analysis())

print("\n=== Group Analysis ===")
groups = report.group_analysis()
print("By school:\n", groups['by_school'])
print("\nBy grade:\n", groups['by_grade'])

print("\n=== Top & Bottom ===")
tb = report.top_and_bottom()
print("Top 5:\n", tb['top_students'])
print("Bottom 5:\n", tb['bottom_students'])

report.plot_report('exploration.png')