import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(
        "data/daily_system_data/win23_subset_zip_grow_tower_side_b_cleaned_interpolated.csv"
    )

    # Convert 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Set 'date' column as index
    df.set_index("date", inplace=True)

    # Plot the columns through time
    fig, axs = plt.subplots(nrows=len(df.columns) - 2, figsize=(15, 18), sharex=True)
    df.plot(subplots=True, ax=axs)

    # Add red lines indicating desired range for initial_ph, initial_ec,
    # and initial_nutrient_solution_volume.
    desired_ranges = {
        "initial_ph": (6, 6.4),
        "initial_ec": (2000, 2400),
        "initial_nutrient_solution_volume": (5, 6),
    }
    for ax, col in zip(axs, df.columns):
        if col in desired_ranges:
            min_val, max_val = desired_ranges[col]
            ax.axhline(y=min_val, color="r", linestyle="--")
            ax.axhline(y=max_val, color="r", linestyle="--")

    plt.tight_layout()

    # Save figure
    plt.savefig(
        "data/daily_system_data_plots/win23_subset_zip_grow_tower_side_b_cleaned_interpolated.png"
    )
