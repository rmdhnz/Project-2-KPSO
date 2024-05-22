import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt

np.random.seed(0)  # Untuk hasil yang dapat direproduksi
original_signal = np.sin(np.linspace(0, 2 * np.pi, 100))
# Menambahkan noise Gaussian
noisy_signal = original_signal + np.random.normal(0, 0.1, original_signal.shape)
# Menerapkan filter Wiener
filtered_signal = wiener(noisy_signal)

plt.figure(figsize=(10, 6))

plt.plot(original_signal, label='Original Signal', linestyle='--', color='green')
plt.plot(noisy_signal, label='Noisy Signal', linestyle='-', color='red')
plt.plot(filtered_signal, label='Filtered Signal (Wiener)', linestyle='-', color='blue')

plt.legend()
plt.title('Wiener Filter Implementation')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()