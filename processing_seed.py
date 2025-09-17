from scipy.io import loadmat, savemat
import numpy as np
import os
import re

# --- 参数设置 ---
FS = 200             # 采样率（Hz）
WINDOW_S = 4         # 每个样本持续时间（秒）
STEP_S = 4           # 样本之间的步长（秒）
BASELINE_S = 60      # 基线校正长度（秒）
RAW_DATA_DIR = r"Preprocessed_EEG" 
OUTPUT_DIR = r"PerSession_4sZScore_62x800"

def load_and_baseline_correct_trials(data_dir, baseline_s=BASELINE_S, fs=FS):
    baseline_len = int(baseline_s * fs)
    try:
        lbl = loadmat(os.path.join(data_dir, "label.mat"))['label'].flatten()
        print(f"Successfully loaded label.mat. The label sequence is: {lbl}")
    except FileNotFoundError:
        print(f"Error: label.mat not found in {data_dir}! Please check the path.")
        return [], [], [], []

    all_files = os.listdir(data_dir)
    data_files = sorted(
        [f for f in all_files if f.lower().endswith('.mat') and f.lower() != 'label.mat'],
        key=lambda f: (int(f.split('_')[0]), f.split('_')[1])
    )

    processed_ids, arrs, labs, subs = [], [], [], []

    for fname in data_files:
        print(f"Processing raw file: {fname}")
        subj = int(fname.split('_')[0])
        mat = loadmat(os.path.join(data_dir, fname))
        
        scenes = sorted(
            [k for k in mat.keys() if not k.startswith('__')],
            key=lambda s: int(re.search(r'(\d+)$', s).group(1))
        )
        
        if len(scenes) != 15:
            print(f"  [Warning] Expected 15 scenes in {fname}, but found {len(scenes)}. Skipping this file.")
            continue

        for scene in scenes:
            trial_num = int(re.search(r'(\d+)$', scene).group(1))
            idx = trial_num - 1
            data = mat[scene]

            if data.shape[0] != 62 or data.shape[1] < baseline_len:
                print(f"  Skipping scene {scene} due to insufficient length ({data.shape[1]} points) for baseline correction.")
                continue

            # --- 【核心修改】Z-Score 标准化 ---
            # 1. 计算基线段的均值和标准差（按通道）
            baseline_data = data[:, :baseline_len]
            baseline_mean = np.mean(baseline_data, axis=1, keepdims=True)
            baseline_std = np.std(baseline_data, axis=1, keepdims=True)
            
            # 2. 为避免除以零，添加一个小的 epsilon
            epsilon = 1e-8
            
            # 3. 应用Z-score标准化到整个trial
            normalized_data = (data - baseline_mean) / (baseline_std + epsilon)
            normalized_data = np.nan_to_num(normalized_data.astype(np.float32))
            # --- 修改结束 ---

            processed_ids.append(f"{subj}_{fname}_{scene}")
            arrs.append(normalized_data) # 使用标准化后的数据
            labs.append(int(lbl[idx]))
            subs.append(subj)

    return processed_ids, arrs, np.array(labs, np.int32), np.array(subs, np.int32)

def segment_trial(trial_data, window_s=WINDOW_S, step_s=STEP_S, fs=FS):
    win_len = window_s * fs
    step_len = step_s * fs
    n_ch, T = trial_data.shape
    segments = []
    
    total_frames = 4
    points_per_frame = fs

    if T < win_len:
        return segments

    for start in range(0, T - win_len + 1, step_len):
        full_segment = trial_data[:, start : start + win_len]
        reshaped_segment = full_segment.reshape(n_ch, total_frames, points_per_frame)
        sample = np.transpose(reshaped_segment, (1, 0, 2))
        segments.append(sample)

    return segments

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ids, trials, labels, subs = load_and_baseline_correct_trials(RAW_DATA_DIR)
    
    if not ids:
        print("No data was processed. Exiting.")
        return

    data_by_session = {}
    for id_str, trial, label, subj in zip(ids, trials, labels, subs):
        parts = id_str.split('_')
        fname_part = next((p for p in parts if p.lower().endswith('.mat')), None)
        if fname_part is None: continue
        
        sess_key = (subj, fname_part)

        if sess_key not in data_by_session:
            data_by_session[sess_key] = {'X': [], 'y': [], 'segs_per_trial': []}
        
        segs = segment_trial(trial)
        num_segs = len(segs)
        data_by_session[sess_key]['segs_per_trial'].append(num_segs)

        if num_segs > 0:
            data_by_session[sess_key]['X'].extend(segs)
            data_by_session[sess_key]['y'].extend([label] * num_segs)

    saved_count = 0
    for (subj, fname), D in sorted(data_by_session.items()):
        if not D['X']:
            print(f"Skipping empty session: {(subj, fname)}")
            continue

        X = np.array(D['X'], np.float32)
        y = np.array(D['y'], np.int32)
        segs_per_trial = np.array(D['segs_per_trial'], np.int32)
        
        out_name = f"subject_{subj:02d}_session_{os.path.splitext(fname)[0]}_seg4x62x200.mat"
        
        savemat(
            os.path.join(OUTPUT_DIR, out_name), 
            {'seg_X': X, 'seg_y': y, 'segs_per_trial': segs_per_trial}, 
            do_compression=True
        )
        print(f"Saved: {out_name}, total segments: {X.shape[0]}")
        saved_count += 1

    print(f"\nPreprocessing complete. Total session files saved: {saved_count}")

if __name__ == '__main__':
    # 定义在主脚本中用到的全局变量
    TOTAL_FRAMES = 4
    N_CHANNELS = 62
    main()