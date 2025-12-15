#!/usr/bin/env python3
# reduce_dataset.py - zmniejsza liczbę linii w sft_dataset.jsonl

import json
import random
import argparse
import os

def reduce_dataset(input_file, output_file, max_lines, random_sample=False):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"Invalid JSON line skipped.")
    
    print(f"Loaded {len(data)} samples.")
    
    if len(data) <= max_lines:
        print(f"Dataset already has {len(data)} lines, no reduction needed.")
        return
    
    if random_sample:
        reduced = random.sample(data, max_lines)
    else:
        reduced = data[:max_lines]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in reduced:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Reduced to {len(reduced)} samples, saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reduce dataset size")
    parser.add_argument("--input", type=str, default="data/sft_dataset.jsonl", help="Input file")
    parser.add_argument("--output", type=str, default="data/sft_dataset_reduced.jsonl", help="Output file")
    parser.add_argument("--max_lines", type=int, default=800, help="Max number of lines")
    parser.add_argument("--random", action='store_true', help="Random sample instead of first N")
    args = parser.parse_args()
    
    reduce_dataset(args.input, args.output, args.max_lines, args.random)