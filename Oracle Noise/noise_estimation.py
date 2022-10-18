import soundfile as sf
import numpy as np
from scipy.fft import fft

# defining certain variables
alpha1 = 0.8
alpha2 = 1
beta = 0.8
gamma = 0.998
eta = 0.7
sigma = 1.3
epsilon = 0.8
NFFT = 1024


# calculation of ratios


def calcRatios(P_cur, N_prev, Fs):
    Psum = Nsum = 0
    i = 0
    freq_res = (Fs/2) / NFFT

    while(i < NFFT and 0 <= i*freq_res <= 1000):
        Psum += P_cur[i]
        Nsum += N_prev[i]
        i += 1
    epsilonL = Psum / Nsum

    Psum = Nsum = 0
    while(i < NFFT and 1000 < i*freq_res <= 3000):
        Psum += P_cur[i]
        Nsum += N_prev[i]
        i += 1
    epsilonM = Psum / Nsum

    Psum = Nsum = 0
    while(i < NFFT and i*freq_res > 3000):
        Psum += P_cur[i]
        Nsum += N_prev[i]
        i += 1
    epsilonH = Psum / Nsum

    return epsilonL, epsilonM, epsilonH


def estimate_noise(y, Fs):
    frame_len = 1024
    hop_size = 256
    num_samples = len(y)

    # initialising variables based on first frame
    y_0 = fft(y[:frame_len], n=NFFT)  # fft of first frame
    N_prev = (1-epsilon)*(np.abs(y_0)**2)
    P_prev = (1-eta)*(np.abs(y_0)**2)
    Pmin = P_prev
    N = N_prev

    prev_window_start = 0
    while(prev_window_start < num_samples):
        y_lamda = fft(y[prev_window_start+hop_size: max(num_samples,prev_window_start+hop_size+frame_len)],
                      n=NFFT)  # fft of current frame
        P_cur = eta*P_prev + (1-eta)*(np.abs(y_lamda)**2)
        epsilonL, epsilonM, epsilonH = calcRatios(
            P_cur=P_cur, N_prev=N_prev, Fs=Fs)

        if(epsilonL < sigma and epsilonM < sigma and epsilonH < sigma):
            N_cur = epsilon*N_prev + (1-epsilon)*(np.abs(y_lamda)**2)
        else:
            freq_res = (Fs/2) / NFFT
            alpha_s = np.array(np.zeros(NFFT))

            for k in range(NFFT):
                if(Pmin[k] < P_cur[k]):
                    Pmin[k] = gamma*Pmin[k] + \
                        ((1-gamma)/(1-beta))*(P_cur[k]-beta*P_prev[k])
                else:
                    Pmin[k] = P_cur[k]

                S_r = P_cur[k]/Pmin[k]

                if(0 <= freq_res*k <= 1000):
                    if(S_r <= 1.3):
                        alpha_s[k] = alpha1
                    else:
                        alpha_s[k] = alpha2

                elif(1000 < freq_res*k <= 3000):
                    if(S_r <= 3):
                        alpha_s[k] = alpha1
                    else:
                        alpha_s[k] = alpha2

                elif(k*freq_res > 3000):
                    if(S_r <= 5):
                        alpha_s[k] = alpha1
                    else:
                        alpha_s[k] = alpha2

            N_cur = alpha_s*N_prev + (1-alpha_s)*(np.abs(y_lamda)**2)

        N = np.vstack((N, N_cur))
        P_prev = P_cur
        prev_window_start += hop_size

    N = N.T
    N = N[0: 513, :]
    N = np.sqrt(N)
    
    return N
    
    
