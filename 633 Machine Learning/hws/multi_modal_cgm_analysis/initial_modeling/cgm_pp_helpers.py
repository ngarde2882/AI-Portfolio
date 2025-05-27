import pandas as pd
import ast
import numpy as np
from datetime import datetime, timedelta

def read_cgm_data(path):
    cgm_data = pd.read_csv(path)
    cgm_data['Breakfast Time'] = pd.to_datetime(cgm_data['Breakfast Time'], errors='coerce')
    cgm_data['Lunch Time'] = pd.to_datetime(cgm_data['Lunch Time'], errors='coerce')

    return cgm_data



def handle_empty_cgm_data(data):
    """
    Drop rows with empty CGM Data and print dropped days for each subject.

    Parameters:
        data (pd.DataFrame): Input DataFrame with 'CGM Data' column.

    Returns:
        pd.DataFrame: DataFrame with rows containing empty CGM lists removed.
    """
    data = data.copy()  # Avoid modifying the original DataFrame

    # Function to check if CGM Data is empty
    def is_cgm_data_empty(cgm_data):
        try:
            parsed_data = ast.literal_eval(cgm_data)
            return isinstance(parsed_data, list) and len(parsed_data) == 0
        except (ValueError, SyntaxError):
            return False

    # Identify rows with empty CGM Data
    empty_cgm_rows = data[data["CGM Data"].apply(is_cgm_data_empty)]

    # Group by Subject ID and Day to print details of dropped rows
    if not empty_cgm_rows.empty:
        grouped = empty_cgm_rows.groupby("Subject ID")["Day"].apply(list)
        for subject_id, days in grouped.items():
            print(f"Subject ID: {subject_id}, Dropped Days: {days} due to missing CGM data (empty list)")

    # Drop rows with empty CGM Data
    data = data[~data["CGM Data"].apply(is_cgm_data_empty)]

    return data

def handle_missing_meal_times(data, method=2):
    """
    Handle missing Breakfast and Lunch Times in the CGM dataset.

    Parameters:
        data (pd.DataFrame): Input DataFrame with missing meal times.
        method (int): Method to handle missing data:
                      1 - Drop rows with missing values.
                      2 - Fill with the average time per person (default).
                      3 - Fill with the global average from all participants.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
        dict: Dictionary of computed averages (None if method=1).
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    data = data.copy()
    averages = None

    def time_only_mean(series):
        """Compute the average time ignoring the date."""
        times = series.dropna().dt.time
        if times.empty:
            return None
        total_seconds = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in times]
        avg_seconds = sum(total_seconds) / len(total_seconds)
        avg_time = (datetime.min + timedelta(seconds=avg_seconds)).time()
        return avg_time

    if method == 1:
        # Drop rows with missing values
        return data.dropna(subset=["Breakfast Time", "Lunch Time"]), None

    elif method == 2:
        # Compute averages per person (HH:mm:ss only)
        averages = data.groupby("Subject ID").agg(
            avg_breakfast=("Breakfast Time", time_only_mean),
            avg_lunch=("Lunch Time", time_only_mean)
        )

        # Fill missing times with subject-specific averages
        def fill_with_avg(row, column, avg_column):
            if pd.isnull(row[column]):
                avg_time = averages.loc[row["Subject ID"], avg_column]
                if avg_time:
                    # Extract the day from CGM Data
                    cgm_data = eval(row["CGM Data"])
                    first_timestamp_str = cgm_data[0][0]  # e.g., '2021-09-19 08:20:00'
                    day = first_timestamp_str.split(" ")[0]
                    # Combine date and time directly
                    return datetime.combine(datetime.strptime(day, "%Y-%m-%d").date(), avg_time)
            return row[column]

        data["Breakfast Time"] = data.apply(
            lambda row: fill_with_avg(row, "Breakfast Time", "avg_breakfast"), axis=1
        )
        data["Lunch Time"] = data.apply(
            lambda row: fill_with_avg(row, "Lunch Time", "avg_lunch"), axis=1
        )

    elif method == 3:
        # Compute global averages (HH:mm:ss only)
        global_avg_breakfast = time_only_mean(data["Breakfast Time"])
        global_avg_lunch = time_only_mean(data["Lunch Time"])

        averages = {
            "avg_breakfast": global_avg_breakfast,
            "avg_lunch": global_avg_lunch
        }

        # Fill missing times with global averages
        def fill_with_global_avg(row, column, avg_time):
            if pd.isnull(row[column]) and avg_time:
                # Extract the day from CGM Data
                cgm_data = eval(row["CGM Data"])
                first_timestamp_str = cgm_data[0][0]  # e.g., '2021-09-19 08:20:00'
                day = first_timestamp_str.split(" ")[0]
                # Combine date and time directly
                return datetime.combine(datetime.strptime(day, "%Y-%m-%d").date(), avg_time)
            return row[column]

        data["Breakfast Time"] = data.apply(
            lambda row: fill_with_global_avg(row, "Breakfast Time", global_avg_breakfast), axis=1
        )
        data["Lunch Time"] = data.apply(
            lambda row: fill_with_global_avg(row, "Lunch Time", global_avg_lunch), axis=1
        )

    return data, averages



def expand_df(data):
    data['Breakfast Time'] = pd.to_datetime(data['Breakfast Time'], errors='coerce')
    data['Lunch Time'] = pd.to_datetime(data['Lunch Time'], errors='coerce')
    data_expanded = []

    # Loop through each row to expand 'CGM Data'
    for _, row in data.iterrows():
        subject_id = row['Subject ID']
        day = row['Day']
        breakfast_time = row['Breakfast Time']
        lunch_time = row['Lunch Time']
        
        # Parse CGM data
        cgm_data = ast.literal_eval(row['CGM Data'])  # Convert string to list of tuples
        
        # Expand each CGM entry into a new row
        for timestamp, cgm_reading in cgm_data:
            data_expanded.append({
                'Subject ID': subject_id,
                'Day': day,
                'Breakfast Time': breakfast_time,
                'Lunch Time': lunch_time,
                'Timestamp': timestamp,
                'CGM Reading': cgm_reading
            })

    # Create a new DataFrame from the expanded data
    expanded_df = pd.DataFrame(data_expanded)

    # Convert `Timestamp` and `CGM Reading` to appropriate types, if necessary
    expanded_df['Timestamp'] = pd.to_datetime(expanded_df['Timestamp'])
    expanded_df['CGM Reading'] = pd.to_numeric(expanded_df['CGM Reading'], errors='coerce')

    return expanded_df

def calculate_and_aggregate_meal_features(df, time_window_hours=2):
    """
    Calculate breakfast and lunch windows, aggregate CGM features, and print subjects with missing data.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing CGM data.
        time_window_hours (int): Number of hours around meal times to define the windows.

    Returns:
        pd.DataFrame: DataFrame with aggregated breakfast and lunch features per Subject ID and Day.
    """
    df = df.copy()

    # Define time windows for breakfast and lunch
    df['Breakfast Window Start'] = df['Breakfast Time'] - pd.Timedelta(hours=time_window_hours)
    df['Breakfast Window End'] = df['Breakfast Time'] + pd.Timedelta(hours=time_window_hours)
    df['Lunch Window Start'] = df['Lunch Time'] - pd.Timedelta(hours=time_window_hours)
    df['Lunch Window End'] = df['Lunch Time'] + pd.Timedelta(hours=time_window_hours)

    # Filter data within breakfast and lunch windows
    df['Breakfast Window'] = df['Timestamp'].between(
        df['Breakfast Window Start'], df['Breakfast Window End']
    )
    df['Lunch Window'] = df['Timestamp'].between(
        df['Lunch Window Start'], df['Lunch Window End']
    )
    breakfast_data = df[df['Breakfast Window']]
    lunch_data = df[df['Lunch Window']]

    # Enhanced feature aggregation for CGM data
    def aggregate_features(data, prefix):
        return data.groupby(['Subject ID', 'Day'])['CGM Reading'].agg(
            **{
                f'{prefix}_mean': 'mean',
                f'{prefix}_std': 'std',
                f'{prefix}_min': 'min',
                f'{prefix}_max': 'max',
                f'{prefix}_median': 'median',
                f'{prefix}_range': lambda x: x.max() - x.min(), 
                f'{prefix}_auc': lambda x: np.trapz(x.values), 
                f'{prefix}_rate_of_change': lambda x: x.diff().mean(), 
                f'{prefix}_skewness': lambda x: x.skew(),
                f'{prefix}_kurtosis': lambda x: x.kurt() 
            }
        ).reset_index()

    # Aggregate features for breakfast and lunch
    breakfast_features = aggregate_features(breakfast_data, 'Breakfast')
    lunch_features = aggregate_features(lunch_data, 'Lunch')

    # Merge breakfast and lunch features
    meal_features = pd.merge(
        breakfast_features, lunch_features,
        on=['Subject ID', 'Day'], how='outer'
    )

    # Identify missing days for breakfast and lunch
    all_days = df[['Subject ID', 'Day']].drop_duplicates()
    days_with_breakfast_data = breakfast_data[['Subject ID', 'Day']].drop_duplicates()
    days_with_lunch_data = lunch_data[['Subject ID', 'Day']].drop_duplicates()

    missing_breakfast_days = all_days.merge(days_with_breakfast_data, on=['Subject ID', 'Day'], how='left', indicator=True)
    missing_breakfast_days = missing_breakfast_days[missing_breakfast_days['_merge'] == 'left_only']

    missing_lunch_days = all_days.merge(days_with_lunch_data, on=['Subject ID', 'Day'], how='left', indicator=True)
    missing_lunch_days = missing_lunch_days[missing_lunch_days['_merge'] == 'left_only']

    # Print missing days for breakfast and lunch
    if not missing_breakfast_days.empty:
        print(f"Subjects with no data around breakfast windows of {time_window_hours} hours:")
        for subject_id, group in missing_breakfast_days.groupby('Subject ID'):
            days = group['Day'].tolist()
            print(f"Subject ID: {subject_id}, Missing Breakfast Days: {days}")

    if not missing_lunch_days.empty:
        print(f"\nSubjects with no data around lunch windows of {time_window_hours} hours:")
        for subject_id, group in missing_lunch_days.groupby('Subject ID'):
            days = group['Day'].tolist()
            print(f"Subject ID: {subject_id}, Missing Lunch Days: {days}")

    return meal_features



class CGMDataPipeline:
    def __init__(self, time_window=2):
        """
        Initialize the CGM Data Pipeline.

        Parameters:
            time_window (int): Number of hours around meal times to define the windows. Default is 2.
        """
        self.time_window = time_window
        self.averages = None  # To store averages (avg_breakfast, avg_lunch)
    def fit_transform(self, df, dropna = True, method = 2):
        """
        Fit the pipeline on training data and transform it.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing CGM data.

        Returns:
            pd.DataFrame: Processed DataFrame with aggregated meal features.
        """
        print("Step 1: Handling empty CGM data...")
        df = handle_empty_cgm_data(df)

        print("\nStep 2: Handling missing meal times (fit)...")
        df, self.averages = handle_missing_meal_times(df, method=method)

        print("\nStep 3: Expanding CGM data...")
        expanded_df = expand_df(df)

        print("\nStep 4: Calculating and aggregating meal features (breakfast and lunch)...")
        processed_df = calculate_and_aggregate_meal_features(
            expanded_df, time_window_hours=self.time_window)

        if dropna:
            print("\nStep 5: Dropping rows with missing features...")
            processed_df = processed_df.dropna()
            print(f"Rows remaining after dropping missing features: {len(processed_df)}")
        return processed_df

