# 文件名: 2_extract_features_final.py
import os
import time
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# 导入 pykalman 库
from pykalman import KalmanFilter

# --- 配置参数 ---
MAX_WORKERS       = 24
FS                = 200
SELECTED_FEATURE  = "PowerSpectrumEntropy"

# --- 网格拓扑配置 ---
GRID_ROWS         = 8
GRID_COLS         = 9
TOTAL_FRAMES      = 4
N_CHANNELS        = 62

# --- 路径配置 ---
INPUT_DIR         = r"PerSession_4sZScore_4x62x200"
OUTPUT_DIR        = f"Feature_{SELECTED_FEATURE}_LDS_Smoothed_4x{GRID_ROWS}x{GRID_COLS}_AllData"

# --- 静态数据 (通道名, 拓扑图) ---
SEED62_CH_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 
    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 
    'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]
CHANNEL_2D_MAP = {
    'AF3': (0, 2), 'FP1': (0, 3), 'FPZ': (0, 4), 'FP2': (0, 5), 'AF4': (0, 6), 'F7': (1, 0), 'F5': (1, 1), 
    'F3': (1, 2), 'F1': (1, 3), 'FZ': (1, 4), 'F2': (1, 5), 'F4': (1, 6), 'F6': (1, 7), 'F8': (1, 8), 
    'FT7': (2, 0), 'FC5': (2, 1), 'FC3': (2, 2), 'FC1': (2, 3), 'FCZ': (2, 4), 'FC2': (2, 5), 'FC4': (2, 6), 
    'FC6': (2, 7), 'FT8': (2, 8), 'T7': (3, 0), 'C5': (3, 1), 'C3': (3, 2), 'C1': (3, 3), 'CZ': (3, 4), 
    'C2': (3, 5), 'C4': (3, 6), 'C6': (3, 7), 'T8': (3, 8), 'TP7': (4, 0), 'CP5': (4, 1), 'CP3': (4, 2), 
    'CP1': (4, 3), 'CPZ': (4, 4), 'CP2': (4, 5), 'CP4': (4, 6), 'CP6': (4, 7), 'TP8': (4, 8), 'P7': (5, 0), 
    'P5': (5, 1), 'P3': (5, 2), 'P1': (5, 3), 'PZ': (5, 4), 'P2': (5, 5), 'P4': (5, 6), 'P6': (5, 7), 'P8': (5, 8), 
    'PO7': (6, 1), 'PO5': (6, 2), 'PO3': (6, 3), 'POZ': (6, 4), 'PO4': (6, 5), 'PO6': (6, 6), 'PO8': (6, 7), 
    'CB1': (7, 2), 'O1': (7, 3), 'OZ': (7, 4), 'O2': (7, 5), 'CB2': (7, 6)
}

# --- 工具与处理函数 ---
def calculate_power_spectrum_entropy(x, fs=FS):
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, len(x)), detrend='linear')
    Pxx = Pxx[Pxx > 1e-10]
    if not len(Pxx): return 0.0
    p = Pxx / Pxx.sum()
    p = p[p > 1e-10]
    return -np.sum(p * np.log(p))

def reshape_to_4x8x9(feat_4x62):
    out = np.zeros((TOTAL_FRAMES, GRID_ROWS, GRID_COLS), dtype=np.float32)
    for ch_idx, ch_name in enumerate(SEED62_CH_NAMES):
        if ch_name in CHANNEL_2D_MAP:
            r, c = CHANNEL_2D_MAP[ch_name]
            out[:, r, c] = feat_4x62[:, ch_idx]
    return out

def apply_lds_smoothing(trial_features, n_em_iter=5):
    original_shape = trial_features.shape
    num_segments = original_shape[0]
    if num_segments <= 1: return trial_features
        
    features_reshaped = trial_features.reshape(num_segments, -1)
    
    kf = KalmanFilter(
        transition_matrices=[1], observation_matrices=[1],
        transition_covariance=1.0, observation_covariance=1.0, # 初始猜测值
        initial_state_mean=0, initial_state_covariance=1
    )
    
    smoothed_features_reshaped = np.zeros_like(features_reshaped)
    
    for i in range(features_reshaped.shape[1]):
        observations = features_reshaped[:, i]
        try:
            kf_learned = kf.em(observations, n_iter=n_em_iter)
            smoothed_states_means, _ = kf_learned.smooth(observations)
            smoothed_features_reshaped[:, i] = smoothed_states_means.flatten()
        except Exception as e:
            print(f"    [Warning] LDS-EM failed for a feature stream: {e}. Using original data for this stream.")
            smoothed_features_reshaped[:, i] = observations
            
    return smoothed_features_reshaped.reshape(original_shape)

def process_file_robust(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Processing: {os.path.basename(file_path)}")
    mat = loadmat(file_path)
    X_raw = mat['seg_X']
    y_raw = mat['seg_y'].flatten()
    segs_per_trial = mat['segs_per_trial'].flatten()

    boundaries = np.cumsum(segs_per_trial)
    full_boundaries = np.concatenate(([0], boundaries))
    num_trials = len(segs_per_trial)
    all_session_features = []

    for i in range(num_trials):
        start_idx, end_idx = int(full_boundaries[i]), int(full_boundaries[i+1])
        if start_idx >= end_idx: continue
        trial_raw_data = X_raw[start_idx:end_idx]
        
        trial_features_list = []
        for j in range(trial_raw_data.shape[0]):
            seg4x62x200 = trial_raw_data[j]
            feats_4x62 = np.zeros((TOTAL_FRAMES, N_CHANNELS), dtype=np.float32)
            for t in range(TOTAL_FRAMES):
                for ch in range(N_CHANNELS):
                    feats_4x62[t, ch] = calculate_power_spectrum_entropy(seg4x62x200[t, ch, :])
            reshaped_feat = reshape_to_4x8x9(feats_4x62)
            trial_features_list.append(reshaped_feat)
        
        if not trial_features_list: continue
        
        trial_features = np.stack(trial_features_list)
        smoothed_trial_features = apply_lds_smoothing(trial_features)
        all_session_features.append(smoothed_trial_features)

    if not all_session_features: return np.array([]), np.array([])
    final_features = np.vstack(all_session_features)
    return final_features, y_raw

# --- 主流程 ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_features_list, all_labels_list = [], []
    file_paths = [os.path.join(INPUT_DIR, f) for f in sorted(os.listdir(INPUT_DIR)) if f.endswith('.mat')]
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(process_file_robust, path): path for path in file_paths}
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                features, labels = future.result()
                if features.size > 0:
                    all_features_list.append(features)
                    all_labels_list.append(labels)
                print(f"  ✓ Completed processing: {os.path.basename(path)}")
            except Exception as exc:
                print(f"  ✗ {os.path.basename(path)} generated an exception: {exc}")
    
    print(f"\nParallel processing finished in {time.time() - start_time:.2f}s.")
    
    if not all_features_list:
        print("No features were processed successfully. Exiting.")
        return

    print("Aggregating results...")
    X_all = np.vstack(all_features_list)
    y_all = np.concatenate(all_labels_list)
    
    print("\n--- Data Aggregation Summary ---")
    print(f"Total samples aggregated: {len(X_all)}")

    output_filename = "all_features_lds_smoothed.mat"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    savemat(output_path, {'features': X_all, 'labels': y_all}, do_compression=True)
    print(f"\nSuccessfully saved all aggregated data to: {output_path}")

if __name__ == '__main__':
    main()