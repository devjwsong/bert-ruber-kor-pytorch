from tqdm import tqdm

import pandas as pd
import argparse
import random
import numpy as np
import json


def run(args):
    args.data_dir = '/'.join(args.raw_data_path.split('/')[:-1])

    # Loading data.
    df = pd.read_excel(args.raw_data_path)

    # Iterating & parsing dialogues.
    total_dials = []
    dial = []
    for r, row in tqdm(df.iterrows()):
        if r > 0:
            tag, utter = row['Unnamed: 0'], row['Unnamed: 1']
            is_start = True if tag == tag else False

            if is_start:
                if len(dial) > 0:
                    total_dials.append(dial)
                dial = []
            
            dial.append(utter)
            
    if len(dial) > 0:
        total_dials.append(dial)

    print(f"The number of total dialogues: {len(total_dials)}")
    num_turns = [len(dial) for dial in total_dials]
    print(f"The maximum number of turns: {np.max(num_turns)}")
    print(f"The average number of turns: {np.mean(num_turns)}")
    print(f"The minimum number of turns: {np.min(num_turns)}")

    # Shuffling & splitting data.
    random.shuffle(total_dials)
    num_train_dials = int(len(total_dials) * args.train_ratio)
    print(f"The number of train dialogues: {num_train_dials}")
    print(f"The number of evaluation dialogues: {len(total_dials) - num_train_dials}")
    train_dials, eval_dials = total_dials[:num_train_dials], total_dials[num_train_dials:]
    with open(f"{args.data_dir}/train_dials.json", 'w') as f:
        json.dump(train_dials, f)
    with open(f"{args.data_dir}/eval_dials.json", 'w') as f:
        json.dump(eval_dials, f)


if __name__=='__main__':
    # Arguments for training.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--raw_data_path', type=str, required=True, help="The raw xlsx data file path.")
    parser.add_argument('--train_ratio', type=float, default=0.9, help="The ratio of train set to total data size.")

    args = parser.parse_args()

    random.seed(args.seed)

    run(args)
    print("FINISHED.")

