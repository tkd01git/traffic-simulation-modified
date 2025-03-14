import pandas as pd

def identify_yellow_lines(df, consecutive_threshold=5):
    """連続して負の値を取る時刻を特定"""
    negative_indices = df['F_lambda2'] < -10
    groups = negative_indices.ne(negative_indices.shift()).cumsum()
    negative_streaks = df.groupby(groups).filter(lambda g: len(g) >= consecutive_threshold and g['F_lambda2'].iloc[0] < 0)
    return negative_streaks['Time'].tolist()

def identify_red_lines(df, speed_threshold):
    """赤線を引く時刻を特定"""
    return df.loc[df['speed'] < speed_threshold, 'Time'].tolist()

def analyze_conditions(file_names, base_path, speed_threshold=50):
    results = {"t_exists_yellow_before": 0, "t_exists_no_yellow_before": 0,
               "t_not_exists_yellow_before": 0, "t_not_exists_no_yellow_before": 0}

    for file_name in file_names:
        file_path = f"{base_path}/{file_name}"

        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            df = df[['Time', 'speed', 'F_lambda2']].dropna()

            # Identify yellow and red lines
            yellow_lines = identify_yellow_lines(df, consecutive_threshold=4)
            red_lines = identify_red_lines(df, speed_threshold)

            has_t = len(red_lines) > 0
            has_yellow_before_t = any(yellow_time < red_lines[0] for yellow_time in yellow_lines) if has_t else False

            if has_t and has_yellow_before_t:
                results["t_exists_yellow_before"] += 1
            elif has_t and not has_yellow_before_t:
                results["t_exists_no_yellow_before"] += 1
            elif not has_t and len(yellow_lines) > 0:
                results["t_not_exists_yellow_before"] += 1
            else:
                results["t_not_exists_no_yellow_before"] += 1

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    return results

# Base path and file names
base_path = "C://Users//YuheiTakada//Downloads//1223traffic-simulation-de"
result_files = [f"result{i}.xlsx" for i in range(1,101)]
reresult_files = [f"reresult{i}.xlsx" for i in range(1,101)]
# Analyze conditions for result and reresult
result_analysis = analyze_conditions(result_files, base_path)
reresult_analysis = analyze_conditions(reresult_files, base_path)

# Display results
print("Result Analysis:")
for condition, count in result_analysis.items():
    print(f"{condition}: {count}")

print("\nReresult Analysis:")
for condition, count in reresult_analysis.items():
    print(f"{condition}: {count}")
