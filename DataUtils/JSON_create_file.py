import os
import json
from pathlib import Path
import argparse
import random

def get_n_frames(video_dir):
    """ Count the number of frames (jpg files) in a video directory """
    return len([f for f in os.listdir(video_dir) if f.endswith('.jpg')])

def create_json_for_dataset(dataset_path, output_json, validation_ratio=0.2, test_ratio=0.1):
    """ Create JSON file for the dataset with training, validation, and test split """
    dataset_path = Path(dataset_path)
    data = {
        "version": "1.0",
        "labels": sorted([d.name for d in dataset_path.iterdir() if d.is_dir()]),
        "database": {}
    }

    for label in data["labels"]:
        class_path = dataset_path / label
        video_folders = [v for v in class_path.iterdir() if v.is_dir()]

        # Shuffle video folders for random splitting
        random.shuffle(video_folders)

        # Determine split points for validation and test
        num_videos = len(video_folders)
        val_split_index = int(num_videos * validation_ratio)
        test_split_index = val_split_index + int(num_videos * test_ratio)

        for i, video_folder in enumerate(video_folders):
            video_name = video_folder.name
            n_frames = get_n_frames(video_folder)

            if i < val_split_index:
                subset = "validation"
            elif i < test_split_index:
                subset = "test"
            else:
                subset = "training"

            data["database"][video_name] = {
                "subset": subset,
                "annotations": {
                    "label": label,
                    "segment": [1, n_frames]
                }
            }

    # Write to the JSON file
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate JSON for a video dataset with training/validation/test split.")
    parser.add_argument('dataset_path', type=str, help="Path to the root directory of the dataset.")
    parser.add_argument('output_json', type=str, help="Path where the output JSON file will be saved.")
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help="Percentage of data to use for validation (default: 0.2)")
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help="Percentage of data to use for testing (default: 0.1)")

    args = parser.parse_args()

    create_json_for_dataset(args.dataset_path, args.output_json, args.validation_ratio, args.test_ratio)
