import numpy as np
import math
import cv2


def psnr(img1, img2):
    p = np.array([])
    for sample in range(img2.shape[0]):
        for band in range(img2.shape[1]):
            mse = np.mean((img1[sample][band] - img2[sample][band]) ** 2)
            if mse < 1.0e-10:
                return 100
            p = np.append(p, 20 * math.log10(np.max(img2[sample][band]) / math.sqrt(mse)))
    return np.mean(p)


def SAM(x_true, x_pred):
    """calculate SAM method in code"""
    dot_sum = np.sum(x_true * x_pred, axis=1)
    norm_true = np.linalg.norm(x_true, axis=1)
    norm_pred = np.linalg.norm(x_pred, axis=1)
    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    return np.mean(res)*180/np.pi


def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    rmse = math.sqrt(mse)
    return rmse


def ERGAS(x_pred, x_turth, d=4):
    """
    calc ergas.
    """
    batches, channels, h, w = x_turth.shape
    ergas = np.array([])
    for sample in range(batches):
        inner_sum = 0
        for channel in range(channels):
            band_img1 = x_pred[sample, channel, :, :, ]
            band_img2 = x_turth[sample, channel, :, :, ]
            rmse_value = rmse(band_img1, band_img2)
            m = np.mean(band_img2)
            inner_sum += (rmse_value / m) ** 2
        mean_sum = inner_sum / channels
        ergas = np.append(ergas, 100 * 0.25 * np.sqrt(mean_sum))
    return np.mean(ergas)


def CC_function1(A, F):
    cc = np.array([])
    for i in range(A.shape[0]):
        for band in range(A.shape[1]):
            # cc.append(np.sum((A[i][band] - np.mean(A[i][band])) * (F[i][band] - np.mean(F[i][band]))) / np.sqrt(
            #     np.sum((A[i][band] - np.mean(A[i][band])) ** 2) * np.sum((F[i][band] - np.mean(F[i][band])) ** 2)))
            Aj = A[i][band] - np.mean(A[i][band])
            Fj = F[i][band] - np.mean(F[i][band])
            inner = np.sum(Aj * Fj)
            mod1 = np.sum((A[i][band] - np.mean(A[i][band])) ** 2)
            mod2 = np.sum((F[i][band] - np.mean(F[i][band])) ** 2)
            mod = np.sqrt(mod1 * mod2)
            cc = np.append(cc, inner / mod)
    for i in range(len(cc)):
        if np.isnan(cc[i]):
            cc[i] = 1
    return np.mean(cc)


def CC_function2(ref, tar):
    # Get dimensions
    batch, bands, rows, cols = tar.shape
    # Initialize output array
    out = np.zeros((batch, bands))
    for b in range(batch):
        # Compute cross correlation for each band
        for i in range(bands):
            tar_tmp = tar[b, i, :, :]
            ref_tmp = ref[b, i, :, :]
            cc = np.corrcoef(tar_tmp.flatten(), ref_tmp.flatten())
            out[b, i] = cc[0, 1]

    return np.mean(out)


def ssim(x_pred, x_truth):
    "The calculation is complicated, use it with caution"
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    bacthes, bands, h, w = x_truth.shape
    x_pred = x_pred.astype(np.float64)
    x_truth = x_truth.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    ssim_map = np.array([])
    for sample in range(bacthes):
        for band in range(bands):
            img1, img2 = x_pred[sample, band, :, :], x_truth[sample, band, :, :]
            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
            ssim_map = np.append(ssim_map, ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                                        (sigma1_sq + sigma2_sq + C2)))
    return np.mean(ssim_map)
