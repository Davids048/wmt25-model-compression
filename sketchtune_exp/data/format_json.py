import json
import os
import argparse

def create_json_data(
    file_path:str,
):
    file_prefix = os.path.splitext(file_path)[0]
    print(f'file_prefix: {file_prefix}')
    
    source_lang = file_prefix.split('-')[-2]
    target_lang = file_prefix.split('-')[-1]
    print(f"source: {source_lang}, target: {target_lang}")

    def extract_lines(file):
        lines = []
        try:
            with open(file, 'r') as f:
                for line in f:
                    lines.append(line)
        except FileNotFoundError as e:
            raise e
        return lines

    source_lines = extract_lines(f"{file_prefix}.{source_lang}")
    target_lines = extract_lines(f"{file_prefix}.{target_lang}")
    
    assert len(source_lines) == len(target_lines), "Source and target have different length!"

    dataset = []
    for s, t in zip(source_lines, target_lines):
        dataset.append({
            "source_sentence": s,
            "translation": t
        })
    
    print("Dumping json...")
    with open(f"{file_prefix}.json", 'w') as f:
        json.dump(dataset, f, indent=4)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--file", "-f", type=str, 
        help="data file to create json with. File extension does not matter.")
    args = parser.parse_args()

    create_json_data(args.file)
