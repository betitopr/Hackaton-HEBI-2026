import subprocess
import json
import numpy as np
import pandas as pd
import cv2
import os

def get_video_metadata(file_path):
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_format', '-show_streams', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    
    # Extract relevant info
    format_info = metadata.get('format', {})
    stream_info = metadata.get('streams', [{}])[0]
    
    # Creation time is often in tags
    tags = format_info.get('tags', {})
    creation_time = tags.get('creation_time', 'N/A')
    
    duration = float(format_info.get('duration', 0))
    fps = eval(stream_info.get('avg_frame_rate', '0/1'))
    
    return {
        'file': os.path.basename(file_path),
        'duration': duration,
        'fps': fps,
        'total_frames': int(stream_info.get('nb_frames', 0)),
        'creation_time': creation_time
    }

# 1. IMU Info
imu_data = np.load('data/40343737_20260313_110600_to_112100_imu.npy')
imu_start_ts = imu_data[0, 0] / 1e9  # Convert ns to seconds
imu_end_ts = imu_data[-1, 0] / 1e9
imu_duration = imu_end_ts - imu_start_ts

print(f"--- IMU Data ---")
print(f"Start Timestamp: {imu_start_ts}")
print(f"Duration: {imu_duration:.4f} seconds")
print(f"Samples: {len(imu_data)}")
print(f"Estimated Sampling Rate: {len(imu_data)/imu_duration:.2f} Hz")

# 2. Video Info
videos = ['data/40343737_20260313_110600_to_112100_left.mp4', 
          'data/40343737_20260313_110600_to_112100_right.mp4']

video_results = []
print(f"\n--- Video Data ---")
for v in videos:
    meta = get_video_metadata(v)
    video_results.append(meta)
    print(f"File: {meta['file']}")
    print(f"  Duration: {meta['duration']:.4f}s")
    print(f"  FPS: {meta['fps']:.2f}")
    print(f"  Frames: {meta['total_frames']}")
    print(f"  Creation Time: {meta['creation_time']}")

# 3. Synchronization Analysis
# Note: Usually the filename itself contains a timestamp
# 40343737_20260313_110600 -> March 13, 2026 at 11:06:00
# Let's check if the duration matches
print(f"\n--- Comparison ---")
for meta in video_results:
    diff = abs(imu_duration - meta['duration'])
    print(f"Difference (IMU vs {meta['file']}): {diff:.4f} seconds")
