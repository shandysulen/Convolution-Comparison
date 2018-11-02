import time
import random
import numpy as np
from scipy import signal

def circular_conv(padded_x, padded_y):
    z = []

    for l in range(N+M-1):
        z_sum = 0
        for n in range(N+M-1):
            if n > l:
                z_sum += padded_x[n] * padded_y[l - n + (N+M-1)]
            else:
                z_sum += padded_x[n] * padded_y[l-n]
        z.append(z_sum)
        
    return z

def pad_set_of_vectors(s, length):
    for i in range(len(s)):
        s[i] = np.pad(s[i], (0, length-1), 'constant', constant_values=(0))

SEQUENCE_SIZE = 1024
DATA_RANGE = 10
NUM_SETS = 5

# Generate random sequences
print(f"Generating {NUM_SETS} sets of x and y sequences each with {SEQUENCE_SIZE} values ranging from 0 to {DATA_RANGE}...\n")
x = [np.random.rand(SEQUENCE_SIZE) * DATA_RANGE for _ in range(NUM_SETS)]
y = [np.random.rand(SEQUENCE_SIZE) * DATA_RANGE for _ in range(NUM_SETS)]

no_padding_x = list(x)
no_padding_y = list(y)

N = len(x[0])
M = len(y[0])

# # Numerical Convolution via FFT
print("Performing Convolution via FFT...\n")
start = time.time_ns()
fft_z = []
for i in range(NUM_SETS):
    fft_z.append(signal.fftconvolve(x[i], y[i]))
fft_time_elapsed = time.time_ns() - start

# Pad vectors x and y to be of length N + M -1 before Circular Convolution
pad_set_of_vectors(x, M)
pad_set_of_vectors(y, N)

# Circular Numerical Convolution
print("Performing Circular Convolution...\n")
start = time.time_ns()
circ_z = []
for i in range(NUM_SETS):
    circ_z.append(circular_conv(x[i], y[i]))
circ_time_elapsed = time.time_ns() - start

# Show that circ_z and fft_z are equivalent
if np.allclose(circ_z, fft_z):
    print(f"""The FFT Convolution Result IS EQUIVALENT to the Circular Convolution Result!
Approximate FFT Convolution Time Elapsed: {fft_time_elapsed} ns
Approximate Circular Convolution Time Elapsed: {circ_time_elapsed} ns
""")
else:
    print("The FFT Convolution Result IS NOT EQUIVALENT to the Circular Convolution Result!")

print("---------------------------------------------\n")

# Linear Convolution (no padding)
print("Performing Linear Convolution...\n")
lin_z = []
for i in range(NUM_SETS):
    lin_z.append(np.convolve(no_padding_x[i], no_padding_y[i]))

# Find difference between lin_z and fft_z
print("Finding error between linear convolution (no padding) and convolution via FFT...")
errors = []
for i in range(NUM_SETS):
    e = 0
    diff = lin_z[i] - fft_z[i]
    for num in diff:
        e += abs(num)
    errors.append(e)

# Output the average error across the five sets
avg_err = sum(errors) / NUM_SETS
print(f"Average error across all {NUM_SETS} sets: {avg_err}")

