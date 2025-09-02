import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot_1 (Xk, k, N) -> None:

    # Define as dimensões da janela do gráfico
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=100)

    # plot
    ax.plot(k[:N//2], np.abs(Xk[:N//2]) + 1e-12)
    ax.set_title("Magnitude Plot")
    ax.set_xlabel("k")
    ax.set_ylabel("|X[k]|")
    ax.grid(True)
    ax.set_xlim(0, 100)          # zoom horizontal
    # opcional: “zoom” vertical
    # ax.set_ylim(0, 5000)

    fig.tight_layout()
    plt.show()

    '''
    # Para usar dois gráficos juntos
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=120, sharex=True, gridspec_kw={'height_ratios':[3,1]})
    '''


def plot_2(xn, xn_rec, N) -> None:
    # Plot do sinal original e do sinal reconstruído
    n = np.arange(N)
    plt.figure()
    plt.grid(True)
    plt.plot(n, xn, 'k--', linewidth=1.5, label='Original')
    plt.plot(n, xn_rec, linewidth=1.2, label='Reconstruído (IFFT)')
    plt.xlabel('n'); plt.ylabel('amplitude'); plt.title('Sinal C com 4096 amostras')
    plt.xlim(0, 1000)
    plt.legend(); plt.tight_layout(); plt.show()
