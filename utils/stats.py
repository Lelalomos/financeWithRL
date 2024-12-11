import pandas as pd

def check_skewed(data:pd.Series):
    mean = data.mean()
    median = data.median()

    skewness = data.skew()

    if mean > median:
        print(f"Right-Skewed, {skewness}")
    elif mean < median:
        print(f"Left-Skewed, {skewness}")
    else:
        print(f"Symmetric, {skewness}")

    





