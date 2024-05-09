from gemtelligence import utils
from gemtelligence import learning
import pandas as pd


def main():
    raw_data = utils.load_client_data(stone_type="sapphire")
    grouped_data = learning.utils.group_rechecks(raw_data, "sapphire")
    sapphire_reference_data = utils.load_sapphires_stones_from_path(
        "data/masterfile_basaltic.h5"
    )

    # Load xlsx file
    df = pd.read_excel("data/basaltic burmese.xlsx", header=None)
    df.columns = ["stone_id", "date", "finalised"]
    df["stone_id"] = pd.to_numeric(df["stone_id"], errors="coerce")

    # Find nans in the mask
    mask = df["stone_id"].isna()
    df = df[~mask]
    
    all_stones_id = grouped_data["val"].index
    assert len(all_stones_id) == len(set(all_stones_id))
    
    # Find the intersection of the stones in the xlsx file and the stones in the grouped data
    basaltic_stones_used_for_training = set(all_stones_id) & set(df["stone_id"])
    
    
    breakpoint()


if __name__ == "__main__":
    main()
