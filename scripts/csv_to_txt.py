#!/usr/bin/env python3
"""
Convert a labeled CSV into individual .txt files for training.
Usage:
  python scripts/csv_to_txt.py \
    --csv path/to/data.csv \
    --text_column text \
    --label_column label \
    --output_dir data/training_data
CSV must have a column for text and a column indicating label (e.g., 'ai' or 'human').
"""
import os
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Split CSV rows into txt files for training")
    parser.add_argument('--csv', required=True, help='Path to input CSV file')
    parser.add_argument('--text_column', required=True, help='Name of text column')
    parser.add_argument('--label_column', required=True, help='Name of label column (values: ai or human)')
    parser.add_argument('--output_dir', default='data/training_data', help='Base output directory for txt files')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # create consistent ai/human subfolders
    ai_folder = os.path.join(args.output_dir, 'ai')
    human_folder = os.path.join(args.output_dir, 'human')
    os.makedirs(ai_folder, exist_ok=True)
    os.makedirs(human_folder, exist_ok=True)

    # Write each row to a text file, mapping labels to ai/human
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Writing text files'):
        raw_lbl = row[args.label_column]
        # treat numeric or string labels
        folder = ai_folder if raw_lbl in (1, 1.0, '1', '1.0', 'ai', 'AI') else human_folder
        content = str(row[args.text_column])
        out_path = os.path.join(folder, f"{idx}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == '__main__':
    main()
