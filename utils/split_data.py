import splitfolders
from pathlib import Path

input_dir = Path("data/curated_data") 
output_dir = Path("data/dataset_model")         

splitfolders.ratio(
    input_dir, 
    output=output_dir, 
    seed=42, 
    ratio=(.7, .15, .15), 
    move=False
)

print(f"procces complete, dataset split on: {output_dir.resolve()}")
