import json
import os
import sys
import logging

logging.basicConfig(filename='logs/make_train.log', level=logging.INFO, encoding='utf-8')

def make_train():
    train_path = 'data/train.jsonl'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        logging.error(f"{train_path} not found.")
        sys.exit(1)
    
    with open('data/train.txt', 'w', encoding='utf-8') as out_f:
        with open(train_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    messages = item['messages']
                    text = ''
                    for msg in messages:
                        role = msg['role']
                        content = msg['content']
                        if role == 'user':
                            text += f'<s>[INST] {content} [/INST] '
                        elif role == 'assistant':
                            text += f'{content} </s>'
                        else:
                            raise ValueError(f"Unknown role: {role}")
                    out_f.write(text + '\n')
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON at line {i+1}: {e}")
                    logging.error(f"Invalid JSON at line {i+1}: {e}")
                    sys.exit(1)
                except ValueError as e:
                    print(f"Error at line {i+1}: {e}")
                    logging.error(f"Error at line {i+1}: {e}")
                    sys.exit(1)
    
    print("train.txt created.")
    logging.info("train.txt created.")

if __name__ == '__main__':
    make_train()