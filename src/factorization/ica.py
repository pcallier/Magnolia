import logging
import os
import numpy as np
from scipy.linalg import toeplitz
from sklearn.decomposition import FastICA
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from ..utils.compare_signals import compare_signals
from ..utils.shift_wavs import shift_signal

def ica(X, sigdim=0, n_components=3):

    if sigdim==0:
        Xm = X
    else:
        Xm = X.T

    ica = FastICA(n_components=n_components)
    S_ = ica.fit_transform(Xm)
    A_ = ica.mixing_
    m_ = ica.mean_

    return A_, m_

def kt_ica(X, num_srcs=2, num_shifts=5, sigdim=0):
    '''
    Koldovsky & Tichavsky data augmentation and hierarchical
    clustering for underdetermined BSS using ICA
    '''
    logger = logging.getLogger(__name__)
    if sigdim!=0:
        assert sigdim == 1, "Only 2-D matrices allowed"
        X = X.T
    num_signals = X.shape[0]
    sig_length = X.shape[1]

    ## Augment
    start_pos = num_shifts
    end_pos = sig_length
    shifted_length = end_pos - start_pos
    shifted_obs = np.zeros((num_signals*num_shifts, shifted_length), dtype=X.dtype)
    num_samples = shifted_obs.shape[1]
    # Fill in observations matrix w/ shifted signals
    for sig_idx in range(num_signals):
        first_col = X[sig_idx, start_pos:(start_pos - num_shifts):-1]
        first_row = X[sig_idx, start_pos:end_pos]
        shifted_obs[(num_shifts*sig_idx):(num_shifts + num_shifts*sig_idx)] = toeplitz(first_col, first_row)
    logger.debug("Absolute sum of shifted sigs: {}".format(
        np.sum(np.abs(shifted_obs),axis=(1,))))
    ## Separate
    fastica = FastICA(n_components=num_signals*num_shifts,
                      max_iter=10000, tol=0.00001, whiten=True)
    components = fastica.fit_transform(shifted_obs.T).T

    ## Cluster
    com_sims = compare_signals(components)
    agg = SpectralClustering(n_clusters=num_srcs, affinity='precomputed')
    agg.fit(com_sims)
    logger.debug(agg.labels_)

    ## Reconstruct
    # Get back into space of the augmented input
    reconstruction_weights = np.zeros((num_srcs, len(components), len(components)),
        dtype=shifted_obs.dtype)
    # Assign weights to clusters
    for i in range(num_srcs):
        logger.debug(fastica.components_)
        logger.debug(np.linalg.pinv(fastica.components_))
        # Why am I doing this?
        reconstruction_weights[i,:,:] = np.linalg.pinv(fastica.components_) @ np.diag(agg.labels_ == i) @ fastica.components_
    logger.debug("Absolute sum of recon weights: {}".format(
        np.sum(np.abs(reconstruction_weights),axis=(1,2))))
    reconstructed = np.zeros((num_srcs,*shifted_obs.shape), dtype=shifted_obs.dtype)
    # reconstructed[:,:,:] = reconstruction_weights @ shifted_obs
    for i in range(num_srcs):
        reconstructed[i,:,:] = reconstruction_weights[i] @ shifted_obs
    logger.debug("Absolute sum of recons before de-lagging: {}".format(
        np.sum(np.abs(reconstructed),axis=(1,2))))
    # Reconstruct sources, accounting for original artificial lags
    # source images = what the microphones heard
    # (reduce 2nd dimension from num_signals*num_lags to num_signals)
    src_imgs_est = np.zeros((num_srcs, num_signals, num_samples), dtype=shifted_obs.dtype)
    for clus in range(num_srcs):
        for lag in range(num_shifts):
            for sig in range(num_signals):
                comp = (sig)*num_shifts + lag
                posn_b = num_samples - num_shifts
                logger.debug("Clus {}, component {}, posn_b {}".format(
                    clus, comp, posn_b))
                rcn_a, rcn_b = (num_shifts-lag), (num_samples-lag)
                logger.debug("Recon from {} to {}".format(rcn_a, rcn_b))
                src_imgs_est[clus, sig, :posn_b] += reconstructed[clus, comp, rcn_a:rcn_b]
                # for n in range(num_samples - num_shifts):
                #     comp = (sig)*num_shifts + lag
                #     posn = n + lag
                #     logger.debug("Clus {}, component {}, n {}".format(
                #        clus, comp, posn))
                #     src_imgs_est[clus, sig, n] = reconstructed[clus, comp, posn]
    logger.debug("Absolute sum of recons before delay and sum: {}".format(
        np.sum(np.abs(src_imgs_est),axis=(1,2))))
    # Delay-and-sum of resulting components
    recon_mono = np.zeros((num_srcs, sig_length), dtype=np.float64)
    logger.debug("\n\tnum_samples: {}\n\tsig_length: {}".format(num_samples, sig_length))
    for i in range(num_srcs):
        # Line up components and sum if there is more than one
        if src_imgs_est.shape[1] > 1:
            recon_mono[i, :num_samples] = np.sum([
                shift_signal(src_imgs_est[i,0],src_imgs_est[i,j])
                for j in range(1,src_imgs_est.shape[1])], axis=0)
        else:
            recon_mono[i, :num_samples] = src_imgs_est[i]
        logger.debug("Absolute sum of signal {}: {}".format(i,
                        np.abs(recon_mono[i]).sum()))

    return recon_mono

if __name__ == "__main__":
    from scipy.io import wavfile
    from IPython.display import display, Audio
    from magnolia.utils.bss_eval import bss_eval_sources
    logging.basicConfig(level=logging.DEBUG)


    #display(Audio("../../data/commchannel/karltiffany-1-left.wav"))

    # Load signals, convert to mono, and add
    base_dir = "data/commchannel"
    # base_dir = "../../data/commchannel"
    fs, wav1 = wavfile.read(os.path.join(base_dir,"karltiffany-1-left.wav"))
    wav1 = wav1[:,0]
    wav1.shape

    _, wav2 = wavfile.read(os.path.join(base_dir,"karltiffany-2-left.wav"))
    wav2.shape
    wav2 = wav2[:,0]
    # wav = wav1 + wav2
    # wav = wav.reshape((1,-1))
    wav = np.stack((wav1,wav2),axis=0)

    #display(Audio(wav, rate=fs))
    separated = kt_ica(wav, num_srcs=2, num_shifts=6)
    print(separated.shape)
    print(wav.shape)
    # separated_new = np.stack((shift_signal(wav[0], separated[0]),
    #                           shift_signal(wav[0], separated[1])))
    #display(Audio(separated[0], rate=fs))
    #display(Audio(separated[1], rate=fs))
    srcs = np.stack((wav1,wav2), axis=0)
    print(np.abs(separated).sum(axis=1))

    sdr, sir, sar, _ = bss_eval_sources(srcs, separated)
    print("SDR: {}\nSIR: {}\nSAR: {}".format(sdr,sir,sar))
