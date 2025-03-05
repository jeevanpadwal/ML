import pandas as pd

def analyze_csv(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Count NaN values per column
    nan_counts = df.isna().sum()
    
    # Count zeros per column (only for numerical columns)
    zero_counts = (df == 0).sum()
    
    # Percentage of missing values
    missing_percentage = (nan_counts / len(df)) * 100
    
    # Percentage of zero values (for numerical columns)
    zero_percentage = (zero_counts / len(df)) * 100
    
    # Data summary
    summary = pd.DataFrame({
        'NaN Count': nan_counts,
        'NaN Percentage': missing_percentage,
        'Zero Count': zero_counts,
        'Zero Percentage': zero_percentage
    })
    
    # Display the summary
    print("Dataset Analysis:")
    print(summary)
    
    # Overall dataset completeness score (1 - % of missing values)
    completeness_score = 100 - (missing_percentage.mean())
    print(f"\nOverall Dataset Completeness: {completeness_score:.2f}%")
    
    return summary

# Example usage
file_path = "D:/programming/Python/ML/6/cleaned_dataset.csv"  # Replace with your actual file path
analyze_csv(file_path)