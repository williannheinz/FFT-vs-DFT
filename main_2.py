import numpy as np
import pandas as pd
import time
import dft_idft 
import graficos as gr
import fft_ifft as ff


# main.py
def main():

    # LEITURA DO SINAL NO CSV ---
    df = pd.read_csv("sinal01_C.csv", header=None)
    # xn = df[0].values                                    # pega a primeira (e única) coluna como array
    xn = df.iloc[0].to_numpy(dtype=float)                  # pega a primeira linha como array
    N = len(xn)                                            # Define N 


    t0 = time.perf_counter()                               # começa a contar

    # APLICA A TRANSFORMADA ---
    #Xk = dft_idft.dft_fun(xn, N)                          # executa a DFT

    #xn_rec = dft_idft.idft_fun(Xk, N)                     # executa a IDFT
    #xn_rec = np.real_if_close(xn_rec, tol=1e-10)
    #xn_rec = np.where(np.abs(xn_rec) < 1e-12, 0.0, xn_rec)        # limpeza de valores muito pequenos


    Xk = ff.fft_fun(xn, N)                                 # executa a FFT                                

    xn_rec = ff.ifft_fun(Xk, N, out_len=len(xn))          # IFFT + corta para o tamanho original



    #t1 = time.perf_counter()                               # termina a contagem
    #print(f"Tempo de execução: {(t1 - t0)*1e3:.4f} ms")    # mostra o tempo de execução em ms
    #print("|Xk|:", np.abs(Xk[:20]))                        # mostra os primeiros 20 valores de |Xk|

    #PREPARA VETOR k PARA GRÁFICOS ---
    k = np.arange(N)

    # CHAMA FUNÇÕES DE GRÁFICOS ---
    #gr.plot_1(Xk, k, N)
    gr.plot_2(xn, xn_rec, N)                                   



if __name__ == "__main__":                                 # executa a main
    main()