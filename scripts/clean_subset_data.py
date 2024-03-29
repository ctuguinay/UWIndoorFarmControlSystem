import pandas as pd
from fractions import Fraction


def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        try:
            # Attempt to convert the string to a fraction
            fraction = Fraction(value)
            return fraction.numerator / fraction.denominator
        except ValueError:
            print(f"Could not convert the string {value} to a float.")


def parse_adjustment_type(adjustment_type):
    fraction = 1
    descriptor = ""
    adjustment_type = adjustment_type.lstrip()
    parts = adjustment_type.split(" ")
    for part in parts:
        if "/" in part:
            fraction *= convert_to_float(part)
        else:
            descriptor += part + " "
    return fraction, descriptor.strip()


if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(
        "data/daily_system_data/win23_subset_zip_grow_tower_side_b_manual_mod.csv"
    )

    # Replace all occurrences of "ADD WATER" with None
    df.replace("ADD WATER", None, inplace=True)

    # Rename columns to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Modify the date column format to add 2023
    df["date"] = df["date"] + "/2023"

    # Extract relevant columns for pH and EC measurements
    pH_adjustments = df[["type_of_ph_adjustment", "amount_of_ph_adjustment_used"]]
    EC_adjustments = df[["type_of_ec_adjustment", "amount_of_ec_adjustment_used"]]

    # Initialize lists to store amounts for each actuator
    pH_down_list = []
    pH_up_list = []
    nutrient_mature_list = []
    nutrient_immature_list = []
    water_list = []

    # Loop through pH adjustments and append pH down and pH up amounts to lists
    for index, row in pH_adjustments.iterrows():
        adjustment_type = row["type_of_ph_adjustment"]
        amount = row["amount_of_ph_adjustment_used"]
        if isinstance(amount, str):
            amount = float(amount.lower().rstrip("ml").rstrip())
        if adjustment_type == "pH Down":
            pH_down_list.append(amount)
            pH_up_list.append(0)
        elif adjustment_type == "pH Up":
            pH_down_list.append(0)
            pH_up_list.append(amount)
        else:
            pH_down_list.append(0)
            pH_up_list.append(0)

    # Loop through EC adjustments and append nutrient and water amounts to lists
    for index, row in EC_adjustments.iterrows():
        adjustment_types = row["type_of_ec_adjustment"]
        if not pd.isna(adjustment_types):
            adjustment_types = adjustment_types.replace(", No Water", "")
            if "," in adjustment_types:
                adjustment_types = adjustment_types.lower().split(",")
            else:
                adjustment_types = [adjustment_types.lower()]
        else:
            adjustment_types = [adjustment_types]
        amounts = [row["amount_of_ec_adjustment_used"]] * len(adjustment_types)
        count = 0
        for adjustment_type, amount in zip(adjustment_types, amounts):
            if isinstance(amount, str):
                amount = convert_to_float(
                    amount.lower().replace("gal", "").replace(" ", "")
                )
            if not pd.isna(adjustment_type):
                if "/" in adjustment_type:
                    fraction, adjustment_type = parse_adjustment_type(adjustment_type)
                    adjusted_amount = amount * fraction
                else:
                    adjusted_amount = amount
                if pd.isna(adjusted_amount):
                    adjusted_amount = 0
            if pd.isna(adjustment_type):
                nutrient_immature_list.append(0)
                nutrient_mature_list.append(0)
                water_list.append(0)
            elif adjustment_type == "mature":
                nutrient_mature_list.append(adjusted_amount)
            elif adjustment_type == "immature":
                nutrient_immature_list.append(adjusted_amount)
            elif adjustment_type == "water":
                water_list.append(adjusted_amount)

        # Check for list imbalances and append None to shorter lists
        nutrient_immature_length = len(nutrient_immature_list)
        nutrient_mature_length = len(nutrient_mature_list)
        water_length = len(water_list)
        max_length = max(
            len(nutrient_immature_list), len(nutrient_mature_list), len(water_list)
        )
        nutrient_immature_list.extend([0] * (max_length - len(nutrient_immature_list)))
        nutrient_mature_list.extend([0] * (max_length - len(nutrient_mature_list)))
        water_list.extend([0] * (max_length - len(water_list)))

    calculated_data_df = pd.DataFrame(
        {
            "pH_down_mL": pH_down_list,
            "pH_up_mL": pH_up_list,
            "nutrient_mature_gallons": nutrient_mature_list,
            "nutrient_immature_gallons": nutrient_immature_list,
            "water_gallons": water_list,
        }
    )

    # Replace None values with 0
    calculated_data_df.fillna(0, inplace=True)

    # Concatenate the original DataFrame and the new DataFrame
    result_df = pd.concat([df, calculated_data_df], axis=1)

    # Remove unnecessary system realization columns from the DataFrame
    result_df.drop(
        columns=[
            "comments",
            "type_of_ph_adjustment",
            "type_of_ec_adjustment",
            "amount_of_ec_adjustment_used",
            "amount_of_ph_adjustment_used",
        ],
        inplace=True,
    )

    # Replaced unmeasured and unknown with None
    result_df.replace("unmeasured ", None, inplace=True)
    result_df.replace("unmeasured", None, inplace=True)
    result_df.replace("unknown", None, inplace=True)

    # Replace any strings that start with unm with None
    result_df.replace("^unm.*", None, regex=True, inplace=True)

    # Save the concatenated DataFrame back to a CSV file
    result_df.to_csv(
        "data/daily_system_data/win23_subset_zip_grow_tower_side_b_cleaned.csv",
        index=False,
        mode="w",
    )
