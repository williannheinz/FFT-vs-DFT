import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fft_ifft as ff  


def energy_time(y: np.ndarray) -> float:

    y = np.asarray(y, float).ravel()
    return float(np.mean(y * y))

def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    return float(np.mean((y - yhat) ** 2))

def keep_mask_by_energy_pairs(X: np.ndarray, keep_frac: float) -> np.ndarray:

    X = np.asarray(X)
    N = X.size
    pairs = []

    # DC (k=0)
    pairs.append(( [0], np.abs(X[0])**2 ))

    # Nyquist (k=N/2) se N par
    if N % 2 == 0:
        nyq = N // 2
        pairs.append(( [nyq], np.abs(X[nyq])**2 ))
        k_start, k_stop = 1, nyq
    else:
        k_start, k_stop = 1, (N - 1)//2 + 1

    # Demais pares (k, N-k)
    for k in range(k_start, k_stop):
        e = np.abs(X[k])**2 + np.abs(X[N - k])**2
        pairs.append(( [k, N - k], e ))

    # ordena por energia decrescente
    pairs.sort(key=lambda t: t[1], reverse=True)

    total_e = sum(e for _, e in pairs)
    keep = np.zeros(N, dtype=bool)
    cum = 0.0
    for idxs, e in pairs:
        if cum / total_e < keep_frac:
            keep[idxs] = True
            cum += e
        else:
            break
    return keep

def top_components(X: np.ndarray, n=6):

    #Retorna os n índices k mais fortes do espectro (lado único)
    N = X.size
    half = N//2 + 1 if N % 2 == 0 else (N + 1)//2
    mag = np.abs(X[:half]).astype(float)
    if half > 0: mag[0] = 0.0                                       # ignora DC
    idx = np.argsort(mag)[-n:][::-1]
    return idx, mag[idx]

# pipeline exigida pelo enunciado ---
def main():

    # ler sinal sem ruído (1ª linha, 4096 amostras)
    df = pd.read_csv("sinal01_C.csv", header=None)
    x_clean = df.iloc[0].to_numpy(dtype=float)
    N = len(x_clean)  # 4096

    # adicionar ruído branco (amplitude 0.1)
    rng = np.random.default_rng(42)                                  # semente p/ reprodutibilidade
    x_noisy = x_clean + 0.1 * rng.standard_normal(N)

    # energia no tempo (como no enunciado)
    En = energy_time(x_noisy)
    print(f"Energia média (sinal ruidoso): {En:.6f}")

    # FFT do sinal ruidoso
    X_noisy = ff.fft_fun(x_noisy, N)

    # identificar componentes mais relevantes (lado único)
    k_top, mag_top = top_components(X_noisy, n=6)
    print("Componentes mais relevantes (k, |X[k]|) no ruidoso:")
    for k, m in zip(k_top, mag_top):
        print(f"  k={int(k):4d} |X|={m:.3f}")

    # Reconstruções para 93% e 96%
    recons = []
    for frac in (0.93, 0.96):
        mask = keep_mask_by_energy_pairs(X_noisy, keep_frac=frac)
        X_keep = np.zeros_like(X_noisy)
        X_keep[mask] = X_noisy[mask]
        x_hat = ff.ifft_fun(X_keep, N, out_len=N).astype(float)
        err = mse(x_clean, x_hat)
        recons.append((frac, x_hat, err))

    # PLOT: dois gráficos, eixo x até 1000 amostras ---
    n = np.arange(N)
    sl = slice(0, 1000)                                                                    # janela a exibir

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), dpi=110, sharex=True)

    # 93%
    frac, x_hat, err = recons[0]
    ax1.grid(True, alpha=0.35)
    ax1.plot(n[sl], x_clean[sl], lw=1.2, color="navy", label="Sinal Original")
    ax1.plot(n[sl], x_hat[sl], lw=1.2, color="tab:green",
        label=f"Reconstruído ({int(frac*100)}% Energia, MSE={err:.4f})")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper left")

    # 96%
    frac, x_hat, err = recons[1]
    ax2.grid(True, alpha=0.35)
    ax2.plot(n[sl], x_clean[sl], lw=1.2, color="navy", label="Sinal Original")
    ax2.plot(n[sl], x_hat[sl], lw=1.2, color="violet",
        label=f"Reconstruído ({int(frac*100)}% Energia, MSE={err:.6e})")
    ax2.set_xlabel("Amostras")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper left")

    fig.suptitle("Comparações com sinais reconstruídos", y=0.98)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
