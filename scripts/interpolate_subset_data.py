import pandas as pd

if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("data/win23_subset_zip_grow_tower_side_b_cleaned.csv")

    # Convert 'date' column to datetime if it's not already
    df["date"] = pd.to_datetime(df["date"])

    # Set 'date' column as index
    df.set_index("date", inplace=True)

    # Interpolate missing values only for specific columns
    columns_to_interpolate = [
        "initial_ec",
        "initial_ph",
        "initial_nutrient_solution_volume",
    ]
    for col in columns_to_interpolate:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method="linear")

    # Reset index to make 'date' a column again
    df.reset_index(inplace=True)

    # Save the DataFrame with interpolated values to a new CSV file
    df.to_csv(
        "data/win23_subset_zip_grow_tower_side_b_cleaned_interpolated.csv", index=False
    )
