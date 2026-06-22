import pandas as pd
import numpy as np

# messy raw data — intentionally dirty
raw_employees = pd.DataFrame({
    'emp_id':   [1, 2, 3, 4, 5, 6, 7, 8, 2, 9],
    'name':     ['  Alice Smith  ', 'BOB JONES', 'charlie brown', 
                 'Diana Prince', None, 'EVE WILSON', 'Frank Miller',
                 'Grace Lee', 'BOB JONES', 'Henry Ford'],
    'age':      [30, 'twenty', 35, 28, 45, None, 32, 29, 'twenty', 41],
    'salary':   ['$95,000', '$72,000', '$88,000', '$65,000', 
                 '$91,000', '$78,000', 'N/A', '$84,000', '$72,000', '$89,000'],
    'dept_id':  [10, 20, 10, 30, 20, 10, 40, 30, 20, None],
    'join_date': ['2020-03-15', '2019-07-22', '2021-01-10', '2022-05-18',
                  '2018-11-30', '2023-02-14', '2020-08-05', '2021-09-12',
                  '2019-07-22', '2022-03-28'],
    'email':    ['alice@co.com', 'bob@co.com', 'charlie@co.com',
                 'diana@co.com', 'eve@co.com', 'invalid-email',
                 'frank@co.com', 'grace@co.com', 'bob@co.com', 'henry@co.com']
})

departments = pd.DataFrame({
    'dept_id':   [10, 20, 30, 40],
    'dept_name': ['Engineering', 'Marketing', 'HR', 'Finance'],
    'location':  ['NYC', 'LA', 'Chicago', 'Boston']
})

class EmployeeDataPipeline:

    def __init__(self, employees: pd.DataFrame, departments: pd.DataFrame):
        self.raw = employees.copy()
        self.departments = departments.copy()
        self.clean = None

    def clean_data(self) -> pd.DataFrame:
        df = self.raw.copy()

        df = df.drop_duplicates(subset=['emp_id'])
        df['name'] = df['name'].str.strip().str.title().fillna('Unknown')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())
        df['salary'] = df['salary'].str.replace('[$,]', '', regex=True)
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        df['salary'] = df['salary'].fillna(df['salary'].mean())
        df['join_date'] = pd.to_datetime(df['join_date'])
        df['email'] = df['email'].where(df['email'].str.contains('@', na=False), None)

        df = df.reset_index(drop=True)
        self.clean = df
        return df

    def enrich(self) -> pd.DataFrame:
        if self.clean is None:
            self.clean_data()

        df = self.clean
        df['years_employed'] = (pd.Timestamp.now() - df['join_date']).dt.days // 365
        df['salary_band'] = np.where(df['salary'] > 85000, 'Senior',
                            np.where(df['salary'] > 70000, 'Mid', 'Junior'))
        df = pd.merge(df, self.departments, on='dept_id', how='left')
        self.clean = df
        return df

    def summary_report(self) -> dict:
        if self.clean is None or 'dept_name' not in self.clean.columns:
            self.enrich()
        
        df = self.clean

        by_dept = df.groupby('dept_name').agg(
            count= ('emp_id', 'count'),
            avg_salary = ('salary', 'mean')
        ).round(2)

        newest = df.loc[df['join_date'].idxmax(), ['name', 'join_date']]
        longest = df.loc[df['years_employed'].idxmax(), ['name', 'years_employed']]

        return({
            'total_employees': len(df),
            'avg_salary': round(df['salary'].mean(), 2),
            'avg_age': round(df['age'].mean(), 2),
            'by_department': by_dept,
            'count_per_band': df['salary_band'].value_counts().to_dict(),
            'newest_hire': newest.to_dict(),
            'Longest serving': longest.to_dict()
        })

    def export(self, filename: str) -> None:  
        if self.clean is None:
            self.enrich()
        self.clean.to_csv(filename, index=False)


#example Usage
pipeline = EmployeeDataPipeline(raw_employees, departments)

clean = pipeline.clean_data()
print("Clean data:")
print(clean[['emp_id', 'name', 'age', 'salary', 'email']].to_string())

enriched = pipeline.enrich()
print("\nEnriched data:")
print(enriched[['name', 'dept_name', 'salary_band', 'years_employed']].to_string())

report = pipeline.summary_report()
print(f"\nTotal employees: {report['total_employees']}")
print(f"Avg salary: ${report['avg_salary']:,.2f}")
print(f"\nBy department:\n{report['by_department']}")
print(f"\nSalary bands:\n{report['count_per_band']}")
print(f"\nNewest hire: {report['newest_hire']}")
print(f"Longest serving: {report['Longest serving']}")

pipeline.export("clean_employees.csv")
print("\nExported to clean_employees.csv")