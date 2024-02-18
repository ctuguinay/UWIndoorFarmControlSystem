import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(
        "data/daily_system_data/win23_subset_zip_grow_tower_side_b_simulated.csv"
    )

    # Drop the first row. Calculating init
    df = df.iloc[1:]

    # Convert 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Set 'date' column as index
    df.set_index("date", inplace=True)

    # Drop final nutrient solution volume
    df.drop(
        columns=[
            "initial_ph",
            "initial_ec",
            "initial_nutrient_solution_volume",
            "final_ec",
            "final_ph",
            "final_nutrient_solution_volume",
        ],
        inplace=True,
    )

    # Plot the columns through time
    fig, axs = plt.subplots(nrows=len(df.columns), figsize=(15, 18), sharex=True)
    df.plot(subplots=True, ax=axs)

    # Add red lines indicating desired range for sim_initial_ph, sim_initial_ec,
    # and sim_initial_nutrient_solution_volume.
    desired_ranges = {
        "sim_initial_ph": (6, 6.4),
        "sim_initial_ec": (2000, 2400),
        "sim_initial_nutrient_solution_volume": (5, 6),
    }
    for ax, col in zip(axs, df.columns):
        if col in desired_ranges:
            min_val, max_val = desired_ranges[col]
            ax.axhline(y=min_val, color="r", linestyle="--")
            ax.axhline(y=max_val, color="r", linestyle="--")

    plt.suptitle("Winter 23 Subset Zip Grow Tower Side B Simulated", y=0.99)
    plt.tight_layout()

    # Save figure
    plt.savefig(
        "data/daily_system_data_plots/win23_subset_zip_grow_tower_side_b_simulated.png"
    )
