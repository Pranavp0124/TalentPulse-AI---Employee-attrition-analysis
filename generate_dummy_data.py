"""
generate_dummy_data.py
======================
Generates realistic HR Employee cohorts adhering to the enterprise IBM HR schema.
Useful for generating new test batches, simulating organizational scenarios,
and testing batch prediction pipelines.
"""

import numpy as np
import pandas as pd


def generate_cohort(n_samples=50, random_seed=42):
    np.random.seed(random_seed)

    departments = ['Research & Development', 'Sales', 'Human Resources']
    dept_weights = [0.65, 0.30, 0.05]

    roles_by_dept = {
        'Research & Development': [
            'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
            'Healthcare Representative', 'Research Director', 'Manager'
        ],
        'Sales': [
            'Sales Executive', 'Sales Representative', 'Manager'
        ],
        'Human Resources': [
            'Human Resources', 'Manager'
        ]
    }

    travel_options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    travel_weights = [0.70, 0.20, 0.10]

    marital_options = ['Married', 'Single', 'Divorced']
    marital_weights = [0.46, 0.32, 0.22]

    records = []
    for i in range(1, n_samples + 1):
        dept = np.random.choice(departments, p=dept_weights)
        role = np.random.choice(roles_by_dept[dept])
        age = int(np.random.randint(22, 60))
        total_working_years = max(1, int(np.random.randint(1, min(40, age - 18))))
        years_at_company = int(np.random.randint(0, min(total_working_years + 1, 25)))
        years_with_curr_manager = int(np.random.randint(0, min(years_at_company + 1, 15)))

        # Baseline salary depends on role and experience
        base_salary = 2500 + (total_working_years * 320) + (1000 if 'Director' in role or 'Manager' in role else 0)
        salary_noise = np.random.randint(-500, 1500)
        monthly_income = int(np.clip(base_salary + salary_noise, 2000, 20000))

        # Risk correlation factors
        overtime = np.random.choice(['Yes', 'No'], p=[0.28, 0.72])
        job_satisfaction = int(np.random.choice([1, 2, 3, 4], p=[0.18, 0.22, 0.35, 0.25]))
        env_satisfaction = int(np.random.choice([1, 2, 3, 4], p=[0.18, 0.22, 0.35, 0.25]))
        work_life_balance = int(np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.45, 0.20]))
        job_involvement = int(np.random.choice([1, 2, 3, 4], p=[0.08, 0.25, 0.50, 0.17]))
        stock_option_level = int(np.random.choice([0, 1, 2, 3], p=[0.43, 0.40, 0.11, 0.06]))
        distance_from_home = int(np.random.randint(1, 30))
        num_companies_worked = int(np.random.randint(0, 10))
        percent_salary_hike = int(np.random.randint(11, 26))

        records.append({
            'EmployeeId': f'EMP-{1000 + i}',
            'Age': age,
            'Department': dept,
            'JobRole': role,
            'MonthlyIncome': monthly_income,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'YearsWithCurrManager': years_with_curr_manager,
            'DistanceFromHome': distance_from_home,
            'StockOptionLevel': stock_option_level,
            'EnvironmentSatisfaction': env_satisfaction,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'JobInvolvement': job_involvement,
            'NumCompaniesWorked': num_companies_worked,
            'PercentSalaryHike': percent_salary_hike,
            'MaritalStatus': np.random.choice(marital_options, p=marital_weights),
            'OverTime': overtime,
            'BusinessTravel': np.random.choice(travel_options, p=travel_weights)
        })

    df = pd.DataFrame(records)
    return df


if __name__ == '__main__':
    cohort_df = generate_cohort(n_samples=50, random_seed=101)
    cohort_df.to_csv("sample_test_cohort.csv", index=False)
    print(f"Sample test cohort with {len(cohort_df)} employees saved to 'sample_test_cohort.csv'.")
