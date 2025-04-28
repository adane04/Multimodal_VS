import subprocess

# path to the dataset splits
path = "/dataset/train_h5/splits/tvsum_tv2heur60_splits.json"

# Loop through split indices
for split_index in range(5):  #
    print(f"Running split {split_index}...")
    
    # Command to run train.py with the split index
    command = ["python", "train.py", "--split_index", str(split_index)]
    
    try:
        # Execute the command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running split {split_index}: {e}")
