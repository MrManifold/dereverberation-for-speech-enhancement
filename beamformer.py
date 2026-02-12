
import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, get_window
from tqdm import tqdm

# -----------------------cd
# User settings
# -----------------------
DATA_GLOB = r"D:\chromedownload\CoreTrainDataset1_v1\Main\Train\Dataset_1\Microphone_Array_Audio\**\*.wav"
OUTPUT_DIR = "./mpdr_wpe_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Algorithm params (paper-recommended defaults)
FS_TARGET = 16000            # process at 16 kHz (paper downsampled recordings to 16 kHz)
N_FFT = 1024
HOP = N_FFT // 4            # 75% overlap -> hop = 256 for 1024 frame
WIN = 'hann'
ALPHA = 0.994               # forgetting factor
EPS = 0.01                  # regularization
DELTA = 2                   # prediction delay (frames)
# WPE filter lengths L can be set different per band — we use single L for simplicity
L = 10
INIT_Rg = 1e-2
INIT_Rh = 1e-4

STEERING_MODE = 'eig'       # 'delay_and_sum' or 'eig' (estimate via eigenvector) or 'provided'
USE_ROI = False             # if True, use ROI constraint (approx by averaging steering within sector)
USE_LCMP = False            # if True, apply LCMP constraint (requires second steering vector)
BETA = 0.01                 # LCMP parameter (1->preserve second source, small->suppress)

# adaptive update step for g if we use LMS fallback
MU_G = 0.05

# -----------------------
# Utilities
# -----------------------
def load_wav_multichannel(path):
    """Load wav; returns y (M, N), sr. Supports files saved as N x M or M x N."""
    data, sr = sf.read(path, always_2d=True)  # shape (frames, channels)
    # SoundFile returns shape (frames, channels)
    data = data.T  # now (M, N)
    return data, sr

def resample_if_needed(x, sr, target_sr):
    if sr == target_sr:
        return x, sr
    # simple resample via scipy.signal.resample is not imported to keep deps low;
    # if resampling needed, user should pre-resample files to 16k or install resampy.
    raise RuntimeError(f"File sample rate {sr} != {target_sr}. Please resample to {target_sr} Hz.")

def stft_multi(x, n_fft=N_FFT, hop=HOP, win=WIN):
    M, N = x.shape
    w = get_window(win, n_fft, fftbins=True)
    # compute STFT for each channel
    f, t, Z0 = stft(x[0,:], fs=1.0, window=w, nperseg=n_fft, noverlap=n_fft-hop, boundary=None, padded=False)
    n_bins = Z0.shape[0]; n_frames = Z0.shape[1]
    X = np.zeros((n_bins, n_frames, M), dtype=np.complex64)
    X[:,:,0] = Z0
    for m in range(1, M):
        _, _, Zm = stft(x[m,:], fs=1.0, window=w, nperseg=n_fft, noverlap=n_fft-hop, boundary=None, padded=False)
        if Zm.shape[1] > n_frames:
            Zm = Zm[:,:n_frames]
        elif Zm.shape[1] < n_frames:
            Zm = np.pad(Zm, ((0,0),(0,n_frames-Zm.shape[1])), mode='constant')
        X[:,:,m] = Zm
    return f, t, X

def istft_single(Y, fs, n_fft=N_FFT, hop=HOP, win=WIN):
    w = get_window(win, n_fft, fftbins=True)
    _, x = istft(Y, fs=fs, window=w, nperseg=n_fft, noverlap=n_fft-hop, input_onesided=True)
    return x

def estimate_steering_eig(X, ref_mic=0):
    """
    Estimate per-frequency steering vector (RTF) from data using principal eigenvector of cross-spectrum.
    X: (F, T, M)
    returns D: (F, M) complex steering vectors (normalized to reference mic)
    """
    F, T, M = X.shape
    D = np.zeros((F, M), dtype=np.complex64)
    for f in range(F):
        # cross-spectrum (sample) (M x M)
        S = (X[f,:,:].conj().T @ X[f,:,:]) / max(1, T)  # shape (M,M)
        # eigen decomposition
        try:
            w, v = np.linalg.eigh(S)
            principal = v[:, np.argmax(w)]
            # normalize to reference mic
            if np.abs(principal[ref_mic]) < 1e-8:
                principal = principal / (np.linalg.norm(principal) + 1e-8)
            else:
                principal = principal / principal[ref_mic]
            D[f,:] = principal
        except np.linalg.LinAlgError:
            D[f,:] = np.ones((M,))/M
    return D

# -----------------------
# Core algorithm
# -----------------------
def mpdr_wpe_bilinear_rsl(X, D, params):
    """
    Implements a per-frequency RLS bilinear MPDR-WPE loop in a practical manner.
    X: (F, T, M)
    D: (F, M) steering vectors
    params: dict of algorithm params
    Returns Xhat (F, T) (single-channel dereverbed estimate)
    """
    F, T, M = X.shape
    alpha = params.get('alpha', ALPHA)
    eps = params.get('eps', EPS)
    delta = params.get('delta', DELTA)
    L = params.get('L', L)
    init_Rg = params.get('init_Rg', INIT_Rg)
    init_Rh = params.get('init_Rh', INIT_Rh)
    mu_g = params.get('mu_g', MU_G)

    # allocate
    Xhat = np.zeros((F, T), dtype=np.complex64)
    # spatial and temporal filters
    h = np.zeros((F, M), dtype=np.complex64)
    for f in range(F):
        h[f,:] = (1.0 / M) * np.ones((M,), dtype=np.complex64)  # delay-and-sum init
    
    g = np.zeros((F, L), dtype=np.complex64)
    # inverse covariances:
    Rg_inv = np.array([np.eye(M, dtype=np.complex64) * (1.0 / init_Rg) for _ in range(F)])
    Rh_inv = np.array([np.eye(L, dtype=np.complex64) * (1.0 / init_Rh) for _ in range(F)])
    # maintain Rg and Rh (covariances) for update clarity
    Rg = np.array([np.eye(M, dtype=np.complex64) * init_Rg for _ in range(F)])
    Rh = np.array([np.eye(L, dtype=np.complex64) * init_Rh for _ in range(F)])

    # buffers for past frames for y and beamformer output z
    y_buf = np.zeros((F, L, M), dtype=np.complex64)   # y_buf[f, l, m] holds y(f, t-delta - l)
    z_buf = np.zeros((F, L), dtype=np.complex64)      # z_buf[f, l] holds Z(f, t-delta - l)

    # processing loop (frame-by-frame)
    for t in range(T):
        for f in range(F):
            y_ft = X[f, t, :]  # (M,)
            # update y_buf with frame (we place most recent at index 0 corresponding to t-delta)
            if t - delta >= 0:
                # shift
                y_buf[f,1:,:] = y_buf[f,:-1,:]
                y_buf[f,0,:] = X[f, t-delta, :]
                # same for z_buf (will be filled after Zh computed)
            # compute current beamformer output Zh = h^H y
            Zh = np.vdot(h[f,:], y_ft)  # scalar
            # compute ybar_h: IL ⊗ h ^H ybar => length L vector
            ybarh = np.zeros((L,), dtype=np.complex64)
            for l in range(L):
                ybarh[l] = np.vdot(h[f,:], y_buf[f, l, :])
            # predicted reverberant part = g^H ybarh
            pred = np.vdot(g[f,:], ybarh)
            # estimate
            Xhat_f_t = Zh - pred
            Xhat[f, t] = Xhat_f_t

            # --- Update temporal filter g (Rh and ph) using RLS-like / Kalman form per paper ---
            # compute lambda (variance) as |Xhat|^2 (with floor)
            lam = max((np.abs(Xhat_f_t)**2).real, 1e-8)
            # update Rh (weighted covariance)
            if t - delta >= 0:
                ybarh_col = ybarh.reshape(-1,1)
                Rh[f] = alpha * Rh[f] + (ybarh_col @ ybarh_col.conj().T) / lam
                # regularize and invert (we could use Woodbury update; here compute inverse for clarity)
                try:
                    Rh_inv[f] = np.linalg.inv(Rh[f] + 1e-6 * np.eye(L))
                except np.linalg.LinAlgError:
                    Rh_inv[f] = np.linalg.pinv(Rh[f] + 1e-6 * np.eye(L))
                # compute ph (weighted correlation) incrementally: approximate via immediate correlation
                # ph_new = alpha*ph_old + ybarh * Zh^*/lam  -> but to keep memory small we do direct incremental update
                # we'll use simple RLS update for g: g = g + Kh * Xhat^*
                numer = Rh_inv[f] @ (ybarh_col)
                denom = alpha + (ybarh_col.conj().T @ (Rh_inv[f] @ ybarh_col))
                kh = (numer / denom).reshape(-1,)
                g[f,:] = g[f,:] + kh * np.conj(Xhat_f_t)

            else:
                # warmup: small LMS update so g does not stay zero
                g[f] = g[f] + mu_g * np.conj(ybarh) * (Xhat_f_t - np.vdot(g[f], ybarh))

            # --- Update spatial filter h (Rg and solution h = Rg_inv b / (b^H Rg_inv b) ) ---
            # Create ybar_g = y_ft - sum_l conj(g_l) * y_buf[f,l,:]
            pred_per_m = np.zeros(M, dtype=np.complex64)
            for l in range(L):
                pred_per_m += np.conj(g[f,l]) * y_buf[f, l, :]
            ybar_g = y_ft - pred_per_m  # M vector
            # update Rg (covariance)
            ybar_g_col = ybar_g.reshape(-1,1)
            Rg[f] = alpha * Rg[f] + (ybar_g_col @ ybar_g_col.conj().T)
            # regularize
            Rg_reg = Rg[f] + eps * np.eye(M, dtype=np.complex64)
            try:
                Rg_inv[f] = np.linalg.inv(Rg_reg)
            except np.linalg.LinAlgError:
                Rg_inv[f] = np.linalg.pinv(Rg_reg)
            # Solve MPDR: h = Rg_inv b / (b^H Rg_inv b)
            b = D[f,:]
            denom = np.vdot(b.conj(), Rg_inv[f] @ b)
            if np.abs(denom) < 1e-12:
                denom = 1e-12
            h[f,:] = (Rg_inv[f] @ b) / denom

            # save current Zh into z_buf for future
            if t - delta >= 0:
                z_buf[f,1:] = z_buf[f,:-1]
                z_buf[f,0] = Zh

    return Xhat

# -----------------------
# Driver
# -----------------------
def process_file(path):
    print("Processing:", path)
    y, sr = load_wav_multichannel(path)  # y: (M, N)
    if sr != FS_TARGET:
        print(f"File sr {sr} != target {FS_TARGET}. Please resample externally.")
        return
    M, N = y.shape

    # compute STFT
    f_bins, t_frames, X = stft_multi(y, n_fft=N_FFT, hop=HOP, win=WIN)
    F, T, M2 = X.shape
    assert M2 == M

    # build steering D
    if STEERING_MODE == 'delay_and_sum':
        D = np.ones((F, M), dtype=np.complex64)
        # normalize to reference microphone (0)
        D = D / np.maximum(np.abs(D[:,[0]]), 1e-8)
    elif STEERING_MODE == 'eig':
        print("Estimating steering via eigenvector method...")
        D = estimate_steering_eig(X, ref_mic=0)
    else:
        raise NotImplementedError("Only 'delay_and_sum' and 'eig' steering modes are implemented in this script.")

    params = {'alpha': ALPHA, 'eps': EPS, 'delta': DELTA, 'L': L, 'init_Rg': INIT_Rg, 'init_Rh': INIT_Rh}

    # Run algorithm
    Xhat = mpdr_wpe_bilinear_rsl(X, D, params)

    # ISTFT to time (single-channel)
    out = istft_single(Xhat, fs=FS_TARGET, n_fft=N_FFT, hop=HOP, win=WIN)
    # scale to original amplitude range (avoiding clipping)
    max_in = np.max(np.abs(y))
    if max_in > 0:
        out = out * (max_in / (np.max(np.abs(out)) + 1e-8))

    # save
    base = os.path.basename(path)
    out_name = os.path.join(OUTPUT_DIR, base.replace('.wav', '_mpdr_wpe.wav'))
    sf.write(out_name, out, FS_TARGET)
    print("Saved:", out_name)

# find files
file_list = glob.glob(DATA_GLOB, recursive=True)
print(f"Found {len(file_list)} wav files.")

for p in tqdm(sorted(file_list)):
    try:
        process_file(p)
    except Exception as e:
        print("Error processing", p, ":", e)
