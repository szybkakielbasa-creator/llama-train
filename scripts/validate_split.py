import json
import os
import random
import sys
import logging

logging.basicConfig(filename='logs/validate_split.log', level=logging.INFO, encoding='utf-8')

def validate_and_split():
    dataset_path = 'data/sft_dataset.jsonl'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        logging.error(f"{dataset_path} not found.")
        sys.exit(1)
    
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'messages' not in item:
                    raise ValueError("Missing 'messages' key")
                for msg in item['messages']:
                    if 'role' not in msg or 'content' not in msg:
                        raise ValueError("Invalid message format")
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {i+1}: {e}")
                logging.error(f"Invalid JSON at line {i+1}: {e}")
                sys.exit(1)
            except ValueError as e:
                print(f"Validation error at line {i+1}: {e}")
                logging.error(f"Validation error at line {i+1}: {e}")
                sys.exit(1)
    
    print(f"Validated {len(data)} samples.")
    logging.info(f"Validated {len(data)} samples.")
    
    random.shuffle(data)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    with open('data/train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    with open('data/val.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print("Split complete: train.jsonl and val.jsonl created.")
    logging.info("Split complete.")

if __name__ == '__main__':
    validate_and_split()