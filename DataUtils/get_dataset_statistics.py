import os
import cv2
import numpy as np
import sys
from pathlib import Path


def analyze_videos_in_folder(folder_path):
    video_durations = []
    video_files = sorted([f for f in folder_path.iterdir() if f.is_file() and f.suffix in ['.avi', '.mp4']])

    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        video_durations.append(duration)
        cap.release()

    return video_durations


def analyze_dataset(dataset_path):
    class_stats = {}
    overall_durations = []

    for class_folder in sorted(dataset_path.iterdir()):
        if class_folder.is_dir():
            class_name = class_folder.name
            print(f"Analyzing class: {class_name}")
            video_durations = analyze_videos_in_folder(class_folder)
            if video_durations:
                class_stats[class_name] = {
                    'num_videos': len(video_durations),
                    'min_duration': np.min(video_durations),
                    'max_duration': np.max(video_durations),
                    'avg_duration': np.mean(video_durations)
                }
                overall_durations.extend(video_durations)

    overall_stats = {
        'num_classes': len(class_stats),
        'num_videos': len(overall_durations),
        'min_duration': np.min(overall_durations) if overall_durations else 0,
        'max_duration': np.max(overall_durations) if overall_durations else 0,
        'avg_duration': np.mean(overall_durations) if overall_durations else 0
    }

    return class_stats, overall_stats


if __name__ == "__main__":
    dataset_path = Path(sys.argv[1])
    class_stats, overall_stats = analyze_dataset(dataset_path)

    print("Class-wise statistics:")
    for class_name, stats in class_stats.items():
        print(f"Class {class_name}:")
        print(f"  Number of videos: {stats['num_videos']}")
        print(f"  Minimum duration: {stats['min_duration']:.2f} seconds")
        print(f"  Maximum duration: {stats['max_duration']:.2f} seconds")
        print(f"  Average duration: {stats['avg_duration']:.2f} seconds")

    print("\nOverall statistics:")
    print(f"Total number of classes: {overall_stats['num_classes']}")
    print(f"Total number of videos: {overall_stats['num_videos']}")
    print(f"Minimum video duration in dataset: {overall_stats['min_duration']:.2f} seconds")
    print(f"Maximum video duration in dataset: {overall_stats['max_duration']:.2f} seconds")
    print(f"Average video duration in dataset: {overall_stats['avg_duration']:.2f} seconds")
