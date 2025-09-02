import numpy as np
import pandas as pd
import time
import dft_idft 
import graficos as gr
import fft_ifft as ff
import energy_mse


# main.py
def main():

    # --- LEITURA DO SINAL NO CSV ---
    df = pd.read_csv("sinal01_B.csv", header=None)
    # xn = df[0].values                                    # pega a primeira (e única) coluna como array
    xn = df.iloc[0].to_numpy(dtype=float)                  # pega a primeira linha como array
    N = len(xn)                                            # Define N 

    # --- APLICA A DFT/IDFT OU FFT/IFFT ---
    _ = ff.fft_fun(xn, N)                                  # warm-up
    soma = 0.0
    for i in range(100):
       
        t0 = time.perf_counter_ns()                                       # começa a contar

        # APLICA A TRANSFORMADA ---
        #Xk = dft_idft.dft_fun(xn, N)                                  # executa a DFT

        #xn_rec = dft_idft.idft_fun(Xk, N)                             # executa a IDFT
        #xn_rec = np.real_if_close(xn_rec, tol=1e-10)
        #xn_rec = np.where(np.abs(xn_rec) < 1e-12, 0.0, xn_rec)        # limpeza de valores muito pequenos


        Xk = ff.fft_fun(xn, N)                                         # executa a FFT                                

        #xn_rec = ff.ifft_fun(Xk, N, out_len=len(xn))                  # IFFT + corta para o tamanho original
        
        dt_ns = time.perf_counter_ns() - t0                            # termina a contagem
        dt_ms = dt_ns * 1e-6
        soma = soma + dt_ms

    
    media = soma / 100
    print(f"Tempo de execução: {(media):.4f} ms")           # mostra o tempo de execução em ms
    #print("|Xk|:", np.abs(Xk[:20]))                        # mostra os primeiros 20 valores de |Xk|

    #k = np.arange(N)                                       # vetor k para os gráficos


    # --- CHAMA FUNÇÕES DE GRÁFICOS ---
    #gr.plot_1(Xk, k, N)
    #gr.plot_2(xn, xn_rec, N)                                   


    # --- EXECUTA ENERGIA E MSE ---
    energy_mse.main()


       

if __name__ == "__main__":                                 # executa a main
    main()