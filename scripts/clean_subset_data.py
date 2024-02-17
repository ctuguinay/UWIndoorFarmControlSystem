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
    df = pd.read_csv("data/win23_subset_zip_grow_tower_side_b_manual_mod.csv")

    # Replace all occurrences of "ADD WATER" with None
    df.replace("ADD WATER", None, inplace=True)

    # Extract relevant columns for pH and EC measurements
    pH_adjustments = df[["Type of pH Adjustment", "Amount of pH Adjustment Used"]]
    EC_adjustments = df[["Type of EC Adjustment", "Amount of EC Adjustment Used"]]

    # Initialize lists to store amounts for each actuator
    pH_down_list = []
    pH_up_list = []
    nutrient_mature_list = []
    nutrient_immature_list = []
    water_list = []

    # Loop through pH adjustments and append pH down and pH up amounts to lists
    for index, row in pH_adjustments.iterrows():
        adjustment_type = row["Type of pH Adjustment"]
        amount = row["Amount of pH Adjustment Used"]
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
        adjustment_types = row["Type of EC Adjustment"]
        if not pd.isna(adjustment_types):
            adjustment_types = adjustment_types.replace(", No Water", "")
            if "," in adjustment_types:
                adjustment_types = adjustment_types.lower().split(",")
            else:
                adjustment_types = [adjustment_types.lower()]
        else:
            adjustment_types = [adjustment_types]
        amounts = [row["Amount of EC Adjustment Used"]] * len(adjustment_types)
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
            nutrient_immature_list.extend(
                [0] * (max_length - len(nutrient_immature_list))
            )
            nutrient_mature_list.extend([0] * (max_length - len(nutrient_mature_list)))
            water_list.extend([0] * (max_length - len(water_list)))

    # Create a new DataFrame for the calculated data
    pH_down_len = len(pH_down_list)
    pH_up_len = len(pH_up_list)
    nutrient_mature_len = len(nutrient_mature_list)
    nutrient_immature_len = len(nutrient_immature_list)
    water_len = len(water_list)

    print("Length of pH Down list:", pH_down_len)
    print("Length of pH Up list:", pH_up_len)
    print("Length of Nutrient Mature list:", nutrient_mature_len)
    print("Length of Nutrient Immature list:", nutrient_immature_len)
    print("Length of Water list:", water_len)

    calculated_data_df = pd.DataFrame(
        {
            "pH Down (mL)": pH_down_list,
            "pH Up (mL)": pH_up_list,
            "Nutrient Mature (gallons)": nutrient_mature_list,
            "Nutrient Immature (gallons)": nutrient_immature_list,
            "Water (gallons)": water_list,
        }
    )

    # Concatenate the original DataFrame and the new DataFrame
    result_df = pd.concat([df, calculated_data_df], axis=1)

    # Remove the 'Comments' column from the DataFrame
    result_df.drop(
        columns=["Comments", "Type of pH Adjustment", "Type of EC Adjustment"],
        inplace=True,
    )

    # Save the concatenated DataFrame back to a CSV file
    result_df.to_csv(
        "data/win23_subset_zip_grow_tower_side_b_cleaned.csv", index=False, mode="w"
    )
