import numpy as np
import pywt
pywt.families(short=False)


# wavelet = pywt.Wavelet('db6')
# coeffs = pywt.wavedec(x['Close'], wavelet, level=7)

# # Plotting the Wavelet Coefficients using imshow
# fig, axes = plt.subplots(len(coeffs), 1, figsize=(10, 10))

# for i, ax in enumerate(axes):
#     im = ax.imshow(np.abs(coeffs[i].reshape(1, -1)), aspect='auto', cmap='jet')
#     ax.set_title(f'Coefficient level {i}')

# # Add a single colorbar on the far right
# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax)

# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()



# Perform the Discrete Wavelet Transform with 'dbN'
# wavelet = pywt.Wavelet('db6')
# coeffs = pywt.wavedec(x['ha_close'], wavelet, level=8)

# # Plotting the Wavelet Coefficients using imshow
# fig, axes = plt.subplots(len(coeffs), 1, figsize=(10, 10))

# for i, ax in enumerate(axes):
#     im = ax.imshow(np.abs(coeffs[i].reshape(1, -1)), aspect='auto', cmap='jet')
#     ax.set_title(f'Coefficient level {i}')

# # Add a single colorbar on the far right
# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax)

# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()


# wavelet = pywt.Wavelet('db6')

def wavelet_transform(data, lvl=8):
    coeff = pywt.wavedec(data, wavelet, mode='symmetric', level=lvl)
    return coeff

# Inverse wavelet transform
def inverse_wavelet_transform(coeffs, lvl=8, clear_levels=4):
    # remove last <clear_levels> finer details
    for i in range(clear_levels):
        coeffs[-i-1] = np.zeros(coeffs[-i-1].shape)
    return pywt.waverec(coeffs, wavelet, mode='symmetric')

def wavelet_denoising(data, wavelet='db4', lvl=8):
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=lvl)
    threshold = np.std(coeffs[-lvl])
    coeffs = [pywt.threshold(c, threshold, mode='soft', substitute=0) for c in coeffs]
    denoised_data = pywt.waverec(coeffs, wavelet)
    #print("Original data: {}".format(data))
    #print("Denoised data using wavelet {}: {}".format(wavelet, denoised_data))
    return denoised_data


def wavelet_denoising2(data, wavelet='db4', lvl=8, clear_levels=4):
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=lvl)
    threshold = np.std(coeffs[0])
    for i in range(clear_levels):
        coeffs[-i-1] = pywt.threshold(coeffs[-i-1], threshold, mode='garrote', substitute=0)     # garrote
    denoised_data = pywt.waverec(coeffs, wavelet)
    #print("Original data: {}".format(data))
    #print("Denoised data using wavelet {}: {}".format(wavelet, denoised_data))
    return denoised_data


def wavelet_denoising_rolling(data, wavelet='db4', lvl=8, clear_levels=4, threshold=None):
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=lvl)
    if threshold is None:
        threshold = np.std(coeffs[0])
    for i in range(clear_levels):
        coeffs[-i-1] = pywt.threshold(coeffs[-i-1], threshold, mode='garrote', substitute=0)     # garrote
    denoised_data = pywt.waverec(coeffs, wavelet)
    #print("Original data: {}".format(data))
    #print("Denoised data using wavelet {}: {}".format(wavelet, denoised_data))
    return denoised_data[-1]