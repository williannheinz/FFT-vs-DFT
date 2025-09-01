import numpy as np

def fft_fun(xn, N=None):
    """
    FFT do sinal xn com zero-padding opcional até N.
    Retorna o vetor complexo Xk (tamanho N).
    """
    x = np.asarray(xn, dtype=float).ravel()               # garante 1D

    if N is None:
        N = len(x)
    if N < len(x):
        raise ValueError("N deve ser >= len(xn)")
    if N > len(x):                                        # zero-padding
        x = np.pad(x, (0, N - len(x)))

    Xk = np.fft.fft(x, n=N)

    return Xk



def ifft_fun(Xk, N=None, out_len=None):
   
    x = np.fft.ifft(np.asarray(Xk), n=N)
    x = np.real_if_close(x, tol=1e-10)                    # remove resíduos imaginários numéricos
    if out_len is not None:
        x = x[:out_len]

    return x