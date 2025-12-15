import subprocess
import os
import sys
import logging

logging.basicConfig(filename='logs/autotest.log', level=logging.INFO, encoding='utf-8')

def autotest():
    model = 'bielik-ssomar'  # assuming the OLLAMA_MODEL
    prompt = "Cześć, kim jesteś?"
    
    try:
        result = subprocess.run(['ollama', 'run', model, prompt], 
                                capture_output=True, text=True, encoding='utf-8', timeout=60)
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            print(f"Error running ollama: {error}")
            logging.error(f"Error running ollama: {error}")
            sys.exit(1)
        
        with open('logs/test_output.txt', 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n\nResponse:\n{output}\n")
        
        print("Test output saved to logs/test_output.txt")
        logging.info("Test completed.")
    
    except subprocess.TimeoutExpired:
        print("Timeout running ollama.")
        logging.error("Timeout running ollama.")
        sys.exit(1)
    except FileNotFoundError:
        print("ollama command not found.")
        logging.error("ollama command not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    autotest()