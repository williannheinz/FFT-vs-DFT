import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# FUNÇÃO DFT ---
def dft_fun(xn, N):

    L = len(xn)
    if N < L:
        raise ValueError("N deve ser sempre maior que L")

    # Preenche o vetor xn com zero até o tamanho N
    xn = np.concatenate([xn, np.zeros(N - L)])

    # Constrói matriz da DFT com for duplo
    X1 = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            Wn = np.exp(-1j * 2 * np.pi * k * n / N)
            X1[k, n] = Wn

    # Multiplicação matricial X = W * x
    Xk = X1 @ xn

    return Xk


# FUNÇÃO IDFT ---
def idft_fun(Xk, N):

    L = len(Xk)
    if N < L:
        raise ValueError("N deve ser sempre maior que L")

    # Preenche o vetor Xk com zero até o tamanho N
    Xk = np.concatenate([Xk, np.zeros(N - L)])

    # Constrói matriz da IDFT com for duplo
    X1 = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for k in range(N):
            Wn = np.exp(1j * 2 * np.pi * k * n / N)
            X1[n, k] = Wn

    # Multiplicação matricial
    xn = (1 / N) * (X1 @ Xk)

    return xn


# --- LEITURA DO SINAL NO CSV ---

df = pd.read_csv("Sinal02.csv", header=None)
xn = df[0].values   # pega a primeira (e única) coluna como array
# Define N (pode ser igual ao tamanho do sinal ou maior)
N = len(xn)  # ou N = 1024, etc.



# Chama função DFT ---
Xk = dft_fun(xn, N)
k = np.arange(N)


# Plot Magnitude
plt.subplot(2, 1, 1)
mag_db = 20 * np.log10(np.abs(Xk[:N//2]) + 1e-12)  # 1e-12 evita log(0)
plt.plot(k[:N//2], mag_db)
plt.ylabel("Magnitude (dB)")
plt.xlabel("k")
plt.title("Magnitude Plot")

# Plot Fase
plt.subplot(2, 1, 2)
plt.plot(k[:N//2], np.unwrap(np.angle(Xk[:N//2])))
plt.xlabel("k")
plt.ylabel("angle(X[k]) (rad)")
plt.title("Phase Plot")

plt.tight_layout()
plt.show()



# Chama função IDFT ---
#IDFT corrigida
xn_rec = idft_fun(Xk, N)
xn_rec = np.real_if_close(xn_rec, tol=1e-10)
# limpeza de valores muito pequenos
xn_rec = np.where(np.abs(xn_rec) < 1e-12, 0.0, xn_rec)

# Plot do sinal original e do sinal reconstruído
n = np.arange(N)
plt.figure()
plt.plot(n, xn, 'k--', linewidth=1.5, label='Original')
plt.plot(n, xn_rec, linewidth=1.2, label='Reconstruído (IDFT)')
plt.xlabel('n'); plt.ylabel('amplitude'); plt.title('Sinal no tempo')
plt.legend(); plt.tight_layout(); plt.show()

