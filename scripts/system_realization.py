import pandas as pd
import numpy as np
from nfoursid.nfoursid import NFourSID

if __name__ == "__main__":
    # Input and Output data columns
    input_columns = [
        "pH_down_mL",
        "pH_up_mL",
        "nutrient_mature_gallons",
        "nutrient_immature_gallons",
        "water_gallons",
    ]
    output_columns = ["initial_ec", "initial_ph", "initial_nutrient_solution_volume"]

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(
        "data/daily_system_data/win23_subset_zip_grow_tower_side_b_cleaned_interpolated.csv"
    )

    # Identify both subspace and system equations using N4SID.
    # Github Repository: https://github.com/spmvg/nfoursid
    # Original Paper:
    # Van Overschee, P., & De Moor, B. (1994). N4SID:
    # Subspace algorithms for the identification of combined deterministic-stochastic systems.
    # Automatica, 30(1), 75-93. https://doi.org/10.1016/0005-1098(94)90230-5
    nfoursid = NFourSID(
        df, input_columns=input_columns, output_columns=output_columns, num_block_rows=1
    )
    nfoursid.subspace_identification()
    state_space_identified, covariance_matrix = nfoursid.system_identification(rank=3)
    A = state_space_identified.a
    B = state_space_identified.b
    C = state_space_identified.c
    D = state_space_identified.d

    # Simulate the identified state space model
    input_dataset = df[input_columns].values
    output_dataset = df[output_columns].values
    initial_x = np.dot(
        np.linalg.inv(C),
        output_dataset[0].reshape(-1, 1) - np.dot(B, input_dataset[0].reshape(-1, 1)),
    )
    initial_output = state_space_identified.output(initial_x)
    state_space_identified._set_x_init(initial_x)
    sim_ec = [initial_output[0][0]]
    sim_ph = [initial_output[1][0]]
    sim_nutrient_solution_volume = [initial_output[2][0]]
    for index, input_data in enumerate(input_dataset[1:]):
        predicted_output = state_space_identified.step(input_data.reshape(-1, 1))
        sim_ec.append(predicted_output[0][0])
        sim_ph.append(predicted_output[1][0])
        sim_nutrient_solution_volume.append(predicted_output[2][0])

    # Create a DataFrame with simulated data
    simulated_output_columns = [
        "sim_initial_ec",
        "sim_initial_ph",
        "sim_initial_nutrient_solution_volume",
    ]
    simulated_df = df.copy()
    simulated_df["sim_initial_ec"] = sim_ec
    simulated_df["sim_initial_ph"] = sim_ph
    simulated_df["sim_initial_nutrient_solution_volume"] = sim_nutrient_solution_volume

    # Reorder columns with desired ranges first
    desired_columns = [
        "sim_initial_ec",
        "sim_initial_ph",
        "sim_initial_nutrient_solution_volume",
    ]
    remaining_columns = [
        col for col in simulated_df.columns if col not in desired_columns
    ]
    reordered_columns = desired_columns + remaining_columns
    simulated_df = simulated_df[reordered_columns]

    # Save simulated DataFrame
    simulated_df.to_csv(
        "data/daily_system_data/win23_subset_zip_grow_tower_side_b_simulated.csv",
        index=False,
    )

    # Save A, B, C, D matrices
    np.savez(
        "data/state_space_model_weights/win23_subset_zip_grow_tower_side_b_simulated.npz",
        A=A,
        B=B,
        C=C,
        D=D,
    )
