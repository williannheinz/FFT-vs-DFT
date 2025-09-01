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



