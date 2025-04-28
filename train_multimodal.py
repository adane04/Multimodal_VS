import sys
import os
from models_train.solver_multimodal import Solver 
from data_loader.data_loader_multimodal2 import get_loader
from configs.configs_multimodal import get_config

def main():
    print("Starting multimodal train.py...")

    split_index = None

    if len(sys.argv) > 2 and sys.argv[1] == '--split_index':
        try:
            split_index = int(sys.argv[2].strip())  # Convert the second argument to an integer
            print(f"Using split index: {split_index}")
        except ValueError:
            print("Error: The split index must be an integer.")
            sys.exit(1)
    else:
        print("Error: '--split_index' argument is missing or incorrectly formatted.")
        sys.exit(1)

    try:
        # Initialize configs 
        config = get_config(mode='train')
        test_config = get_config(mode='test')
       
        print(f"Video input size: {config.video_input_size}")
        print(f"Text input size: {config.text_input_size}")
        print(f"Hidden size: {config.hidden_size}")
        
        print(test_config)

        train_loader = get_loader(
            name='tvsum',  # Base dataset name (must match the prefix in the splits file)
            mode='train',
            split_index=split_index,
            batch_size=config.batch_size,
            shuffle=True
        )

        test_loader = get_loader(
            name='tvsum',  # Same as train
            mode='test',
            split_index=split_index,
            batch_size=test_config.batch_size,
            shuffle=False
        )

        # Build and train the model
        solver = Solver(config, train_loader, test_loader)
        solver.build()

        solver.evaluate(-1)  # Evaluates using initial random weights

        solver.train()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
     # for running on single split, run this command: python train_multimodal.py --split_index 0
# for running on 5 splits, run: python run_splits.py