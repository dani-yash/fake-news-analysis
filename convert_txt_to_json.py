import os
import json

def convert_txt_to_json(txt_file_path, json_file_path):
    data = {}
    with open(txt_file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)
            if key in ['in_degree_distribution', 'out_degree_distribution', 'betweenness_centrality', 'closeness_centrality']:
                value = json.loads(value.replace("'", "\""))
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            data[key] = value

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    base_dirs = ['analysis_results15', 'analysis_results16']
    for base_dir in base_dirs:
        for label in ['true', 'false', 'unverified', 'non-rumor']:
            label_dir = os.path.join(base_dir, label)
            for txt_file in os.listdir(label_dir):
                if txt_file.endswith('_analysis.txt'):
                    txt_file_path = os.path.join(label_dir, txt_file)
                    json_file_path = txt_file_path.replace('_analysis.txt', '_graphs_analysis.json')
                    convert_txt_to_json(txt_file_path, json_file_path)
                    print(f"Converted {txt_file_path} to {json_file_path}")