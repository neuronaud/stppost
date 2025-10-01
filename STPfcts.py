import numpy as np
from numba import jit
from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

def OnePreNpostSTP(num_recipients=10,num_spikes=10,alphapre=.1,alphapost=.9, sender_period=10, t_sim = 1000, Poisson=False, STF=False,baseline=0,isivec=[],sender_times=[]):

  # Parameters for the Integrate and Fire Neuron
  tau_m = 20.0  # Membrane time constant (ms)
  V_rest = -65.0  # Resting potential (mV)
  V_threshold = -50.0  # Firing threshold (mV)
  V_reset = -65.0  # Reset potential (mV)
  R_m = 20.0  # Membrane resistance (MOhms)

  # Simulation parameters
  dt = 0.1  # Time step (ms)
  time = np.arange(0, t_sim, dt)
  num_steps = len(time)

  # Parameters for the sender neuron
  if Poisson==False:
    train_times = np.arange(0, num_spikes * sender_period, sender_period)
    sender_times = np.append(train_times,train_times+num_spikes * sender_period+200)
    sender_times = np.append(sender_times,train_times+2*(num_spikes * sender_period+200))
    isivec = np.diff(sender_times)
  elif len(isivec)==0:
    isivec = np.random.exponential(scale=sender_period/dt, size=num_spikes*3-1)
    sender_times = 10+np.cumsum(isivec)


  # Parameters for the STP distribution
  taus = [9,10,11]


  # Parameters for the recipient neurons
  tau_syn = 5.0  # Synaptic time constant (ms)
  syn_weight = 200.0  # Synaptic weight (nA)

  weights = np.zeros((num_recipients,num_spikes*3-1))
  allkernels = []
  storeNet = np.zeros((num_recipients))
  for j in range(num_recipients):
    if STF == True:
      apre = uniform.rvs(loc=0,scale=6,size=3)*alphapre # from 0 to 6, so STF only
      bpost = uniform.rvs(loc=alphapost,scale=1-alphapost,size=3) # from alphapost to 1 , so if alphapost >0 : when STF it only less STF  (if alpha post <0, it allows for some STF turned STD)
      baselinefac = baseline#uniform.rvs(loc=-6,scale=12,size=1)
    else:
      apre = uniform.rvs(loc=-6,scale=6,size=3)*alphapre # from -6 to 0, so STD only
      bpost = uniform.rvs(loc=1,scale=5*alphapost,size=3) # from 1 to 5*alphapost+1 , so when STD it only makes more STD with post
      baselinefac = baseline#uniform.rvs(loc=-6,scale=12,size=1)
    model = easySRP(mu_amps=[taus[0]*apre[0]*bpost[0], taus[1]*apre[1]*bpost[1], taus[2]*apre[2]*bpost[2]],mu_baseline=baselinefac)
    means, efficacies =  model.run_ISIvec(isivec, ntrials=1, fast=True, return_all=False)
    weights[j,:] = means*syn_weight
    allkernels.append(model.mu_kernel+baselinefac)
    storeNet[j] = np.mean(apre*bpost*taus)





  # Initialize membrane potentials and synaptic currents
  V_sender = np.full(num_steps, V_rest)
  V_recipients = np.full((num_recipients, num_steps), V_rest)
  I_synaptic = np.zeros((num_recipients, num_steps))

  # Simulation loop
  sender_spike_indices = (sender_times / dt).astype(int)
  sender_spike_indices = sender_spike_indices[sender_spike_indices < num_steps]
  weight_index = 0

  for i in range(1, num_steps):
      # Sender neuron (simply fires at predetermined times)
      if i in sender_spike_indices:
          V_sender[i] = V_threshold + 1e-6 # Indicate a spike
          # Deliver synaptic current to recipients
          for j in range(num_recipients):
              I_synaptic[j, i] += weights[j,weight_index] / tau_syn
          weight_index +=1

      # Update synaptic current for recipients
      #I_synaptic[:, i] = I_synaptic[:, i-1] * np.exp(-dt / tau_syn)

      # Update membrane potential for recipients (Integrate and Fire model)
      dV_recipients = (-(V_recipients[:, i-1] - V_rest) + R_m * I_synaptic[:, i]) / tau_m * dt
      V_recipients[:, i] = V_recipients[:, i-1] + dV_recipients

  return V_recipients,allkernels, V_sender,time, storeNet, isivec, sender_times


def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False,
                     axis=-1):
    """Spectral Entropy.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sf : float
        Sampling frequency, in Hz.
    method : str
        Spectral estimation method:

        * ``'fft'`` : Fourier Transform (:py:func:`scipy.signal.periodogram`)
        * ``'welch'`` : Welch periodogram (:py:func:`scipy.signal.welch`)
    nperseg : int or None
        Length of each FFT segment for Welch method.
        If None (default), uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
    axis : int
        The axis along which the entropy is calculated. Default is -1 (last).

    Returns
    -------
    se : float
        Spectral Entropy

    Notes
    -----
    Spectral Entropy is defined to be the Shannon entropy of the power
    spectral density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]

    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.

    References
    ----------
    - Inouye, T. et al. (1991). Quantification of EEG irregularity by
      use of the entropy of the power spectrum. Electroencephalography
      and clinical neurophysiology, 79(3), 204-210.

    - https://en.wikipedia.org/wiki/Spectral_density

    - https://en.wikipedia.org/wiki/Welch%27s_method

    Examples
    --------
    Spectral entropy of a pure sine using FFT

    >>> import numpy as np
    >>> import entropy as ent
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ent.spectral_entropy(x, sf, method='fft'), 2)
    0.0

    Spectral entropy of a random signal using Welch's method

    >>> np.random.seed(42)
    >>> x = np.random.rand(3000)
    >>> ent.spectral_entropy(x, sf=100, method='welch')
    6.980045662371389

    Normalized spectral entropy

    >>> ent.spectral_entropy(x, sf=100, method='welch', normalize=True)
    0.9955526198316071

    Normalized spectral entropy of 2D data

    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> np.round(ent.spectral_entropy(x, sf=100, normalize=True), 4)
    array([0.9464, 0.9428, 0.9431, 0.9417])

    Fractional Gaussian noise with H = 0.5

    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ent.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9505

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ent.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.8477

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ent.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9248
    """
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, sf, axis=axis)
    elif method == 'welch':
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se

def get_stimvec(ISIvec, dt=0.1, null=0, extra=10):
    """
    Generates a binary stimulation vector from a vector with ISI intervals
    :param ISIvec: ISI interval vector (in ms)
    :param dt: timestep (ms)
    :param null: 0s in front of the vector (in ms)
    :param extra: 0s after the last stimulus (in ms)
    :return: binary stim vector
    """

    ISIindex = np.cumsum(
        np.round(np.array([i if i == 0 else i - dt for i in ISIvec]) / dt, 1)
    )
    # ISI times accounting for base zero-indexing

    spktr = np.array(
        [0] * int(null / dt)
        + [
            1 if i in ISIindex.astype(int) else 0
            for i in np.arange(int(sum(ISIvec) / dt + extra / dt))
        ]
    ).astype(bool)

    # Remove redundant dimension
    return spktr


from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import lfilter


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# HELPER FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _refactor_gamma_parameters(mu, sigma):
    """
    Refactor gamma parameters from mean / std to shape / scale
    :param mu: mean parameter as given by the SRP model
    :param sigma: standard deviation parameter as given by the SRP model
    :return: shape and scale parameters
    """
    return (mu ** 2 / sigma ** 2), (sigma ** 2 / mu)


def _sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def _convolve_spiketrain_with_kernel(spiketrain, kernel):
    # add 1 timestep to each spiketime, because efficacy increases AFTER a synaptic release)
    spktr = np.roll(spiketrain, 1)
    spktr[0] = 0  # In case last entry of the spiketrain was a spike
    return lfilter(kernel, 1, spktr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# EFFICIENCY KERNELS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EfficiencyKernel(ABC):

    """ Abstract Base class for a synaptic efficacy kernel"""

    def __init__(self, T=None, dt=0.1):

        self.T = T  # Length of the kernel in ms
        self.dt = dt  # timestep
        self.kernel = np.zeros(int(T / dt))

    @abstractmethod
    def _construct_kernel(self, *args):
        pass


class GaussianKernel(EfficiencyKernel):

    """
    An efficacy kernel from a sum of an arbitrary number of normalized gaussians
    """

    def __init__(self, amps, mus, sigmas, T=None, dt=0.1):
        """
        :param amps: list of floats: amplitudes.
        :param mus: list of floats: means.
        :param sigmas: list or 1: std deviations.
        :param T: length of synaptic kernel in ms.
        :param dt: timestep in ms. defaults to 0.1 ms.
        """

        # Check number of gaussians that make up the kernel
        assert (
            np.size(amps) == np.size(mus) == np.size(sigmas)
        ), "Unequal number of parameters"

        # Default T to largest mean + 5x largest std
        if T is None:
            T = np.max(mus) + 5 * np.max(sigmas)

        # Convert to 1D numpy arrays
        amps = np.atleast_1d(amps)
        mus = np.atleast_1d(mus)
        sigmas = np.atleast_1d(sigmas)

        super().__init__(T, dt)

        self._construct_kernel(amps, mus, sigmas)

    def _construct_kernel(self, amps, mus, sigmas):
        """ constructs the efficacy kernel """

        t = np.arange(0, self.T, self.dt)
        L = len(t)
        n = np.size(amps)  # number of gaussians

        self._all_gaussians = np.zeros((n, L))
        self.kernel = np.zeros(L)

        for i in range(n):
            a = amps[i]
            mu = mus[i]
            sig = sigmas[i]

            self._all_gaussians[i, :] = (
                a
                * np.exp(-((t - mu) ** 2) / 2 / sig ** 2)
                / np.sqrt(2 * np.pi * sig ** 2)
            )

        self.kernel = self._all_gaussians.sum(0)


class ExponentialKernel(EfficiencyKernel):

    """
    An efficacy kernel from a sum of an arbitrary number of Exponential decays
    """

    def __init__(self, taus, amps=None, T=None, dt=0.1):
        """
        :param taus: list of floats: exponential decays.
        :param amps: list of floats: amplitudes (optional, defaults to 1)
        :param T: length of synaptic kernel in ms.
        :param dt: timestep in ms. defaults to 0.1 ms.
        """

        if amps is None:
            amps = np.array([1] * np.size(taus))
        else:
            # Check number of exponentials that make up the kernel
            assert np.size(taus) == np.size(amps), "Unequal number of parameters"

        # Convert to 1D numpy arrays
        taus = np.atleast_1d(taus)
        amps = np.atleast_1d(amps)

        # Default T to 10x largest time constant
        if T is None:
            T = 10 * np.max(taus)

        super().__init__(T, dt)

        self._construct_kernel(amps, taus)

    def _construct_kernel(self, amps, taus):
        """ constructs the efficacy kernel """

        t = np.arange(0, self.T, self.dt)
        L = len(t)
        n = np.size(amps)  # number of gaussians

        self._all_exponentials = np.zeros((n, L))
        self.kernel = np.zeros(L)

        for i in range(n):
            tau = taus[i]
            a = amps[i]

            # set amplitude to a/tau to normalize integrals of all kernels
            self._all_exponentials[i, :] = a / tau * np.exp(-t / tau)

        self.kernel = self._all_exponentials.sum(0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# SRP MODEL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DetSRP:
    def __init__(self, mu_kernel, mu_baseline, mu_scale=None, nlin=_sigmoid, dt=0.1):
        """
        Initialization method for the deterministic SRP model.

        :param kernel: Numpy Array or instance of `EfficiencyKernel`. Synaptic STP kernel.
        :param baseline: Float. Baseline parameter
        :param nlin: nonlinear function. defaults to sigmoid function
        """

        self.dt = dt
        self.nlin = nlin
        self.mu_baseline = mu_baseline

        if isinstance(mu_kernel, EfficiencyKernel):
            assert (
                self.dt == mu_kernel.dt
            ), "Timestep of model and efficacy kernel do not match"
            self.mu_kernel = mu_kernel.kernel
        else:
            self.mu_kernel = np.array(mu_kernel)

        # If no mean scaling parameter is given, assume normalized amplitudes
        if mu_scale is None:
            mu_scale = 1 / self.nlin(self.mu_baseline)
        self.mu_scale = mu_scale

    def run_spiketrain(self, spiketrain, return_all=False):

        filtered_spiketrain = self.mu_baseline + _convolve_spiketrain_with_kernel(
            spiketrain, self.mu_kernel
        )
        nonlinear_readout = self.nlin(filtered_spiketrain) * self.mu_scale
        efficacytrain = nonlinear_readout * spiketrain
        efficacies = efficacytrain[np.where(spiketrain == 1)[0]]

        if return_all:
            return {
                "filtered_spiketrain": filtered_spiketrain,
                "nonlinear_readout": nonlinear_readout,
                "efficacytrain": efficacytrain,
                "efficacies": efficacies,
            }

        else:
            return efficacytrain, efficacies

    def run_ISIvec(self, isivec, **kwargs):
        """
        Returns efficacies given a vector of inter-stimulus-intervals.

        :param isivec: ISI vector
        :param kwargs: Keyword arguments to be passed to 'run' and 'get_stimvec'
        :return: return from `run` method
        """

        spiketrain = get_stimvec(isivec, **kwargs)
        return self.run_spiketrain(spiketrain, **kwargs)


class ProbSRP(DetSRP):
    def __init__(
        self,
        mu_kernel,
        mu_baseline,
        sigma_kernel,
        sigma_baseline,
        mu_scale=None,
        sigma_scale=None,
        **kwargs
    ):
        """
        Initialization method for the probabilistic SRP model.

        :param mu_kernel: Numpy Array or instance of `EfficiencyKernel`. Mean kernel.
        :param mu_baseline: Float. Mean Baseline parameter
        :param sigma_kernel: Numpy Array or instance of `EfficiencyKernel`. Variance kernel.
        :param sigma_baseline: Float. Variance Baseline parameter
        :param sigma_scale: Scaling parameter for the variance kernel
        :param **kwargs: Keyword arguments to be passed to constructor method of `DetSRP`
        """

        super().__init__(mu_kernel, mu_baseline, mu_scale, **kwargs)

        # If not provided, set sigma kernel to equal the mean kernel
        if sigma_kernel is None:
            self.sigma_kernel = self.mu_kernel
            self.sigma_baseline = self.mu_baseline
        else:
            if isinstance(sigma_kernel, EfficiencyKernel):
                assert (
                    self.dt == sigma_kernel.dt
                ), "Timestep of model and variance kernel do not match"
                self.sigma_kernel = sigma_kernel.kernel
            else:
                self.sigma_kernel = np.array(sigma_kernel)

            self.sigma_baseline = sigma_baseline

        # If no sigma scaling parameter is given, assume normalized amplitudes
        if sigma_scale is None:
            sigma_scale = 1 / self.nlin(self.sigma_baseline)
        self.sigma_scale = sigma_scale

    def run_spiketrain(self, spiketrain, ntrials=1):

        spiketimes = np.where(spiketrain == 1)[0]
        efficacytrains = np.zeros((ntrials, len(spiketrain)))

        mean = (
            self.nlin(
                self.mu_baseline
                + _convolve_spiketrain_with_kernel(spiketrain, self.mu_kernel)
            )
            * spiketrain
            * self.mu_scale
        )
        sigma = (
            self.nlin(
                self.sigma_baseline
                + _convolve_spiketrain_with_kernel(spiketrain, self.sigma_kernel)
            )
            * spiketrain
            * self.sigma_scale
        )

        # Sampling from gamma distribution
        efficacies = self._sample(mean[spiketimes], sigma[spiketimes], ntrials)
        efficacytrains[:, spiketimes] = efficacies

        return mean[spiketimes], sigma[spiketimes], efficacies, efficacytrains

    def _sample(self, mean, sigma, ntrials):
        """
        Samples `ntrials` response amplitudes from a gamma distribution given mean and sigma
        """

        return np.random.gamma(
            *_refactor_gamma_parameters(mean, sigma),
            size=(ntrials, len(np.atleast_1d(mean))),
        )


class ExpSRP(ProbSRP):
    """
    SRP model in which mu and sigma kernels are parameterized by a set of amplitudes and respective exponential
    decay time constants.

    This implementation of the SRP model is used for statistical inference of parameters and can be integrated
    between spikes for efficient numerical implementation.
    """

    def __init__(
        self,
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale=None,
        sigma_scale=None,
        **kwargs
    ):

        # Convert to at least 1D arrays
        mu_taus = np.atleast_1d(mu_taus)
        mu_amps = np.atleast_1d(mu_amps)
        sigma_taus = np.atleast_1d(sigma_taus)
        sigma_amps = np.atleast_1d(sigma_amps)

        # Construct mu kernel and sigma kernel from amplitudes and taus
        mu_kernel = ExponentialKernel(mu_taus, mu_amps, **kwargs)
        sigma_kernel = ExponentialKernel(sigma_taus, sigma_amps, **kwargs)

        # Construct with kernel objects
        super().__init__(
            mu_kernel, mu_baseline, sigma_kernel, sigma_baseline, mu_scale, sigma_scale
        )

        # Save amps and taus for version that is integrated between spikes
        self._mu_taus = np.array(mu_taus)
        self._sigma_taus = np.array(sigma_taus)

        # normalize amplitudes by time constant to ensure equal integrals of exponentials
        self._mu_amps = np.array(mu_amps) / self._mu_taus
        self._sigma_amps = np.array(sigma_amps) / self._sigma_taus

        # number of exp decays
        self._nexp_mu = len(self._mu_amps)
        self._nexp_sigma = len(self._sigma_amps)

    def run_ISIvec(self, isivec, ntrials=1, fast=True, return_all=False, **kwargs):
        """
        Overrides the `run_ISIvec` method because the SRP model with
        exponential decays can be integrated between spikes,
        therefore speeding up computation in some cases
        (if ISIs are large, i.e. presynaptic spikes are sparse)

        :return: efficacies
        """

        # Fast evaluation (integrate between spikes)
        if fast:

            state_mu = np.zeros(self._nexp_mu)  # assume kernels have decayed to zero
            state_sigma = np.zeros(
                self._nexp_sigma
            )  # assume kernels have decayed to zero

            means = []
            sigmas = []

            for spike, dt in enumerate(isivec):

                if spike > 0:
                    # At the first spike, read out baseline efficacy
                    # At every following spike, integrate over the ISI and then read out efficacy
                    state_mu = (state_mu + self._mu_amps) * np.exp(-dt / self._mu_taus)
                    state_sigma = (state_sigma + self._sigma_amps) * np.exp(
                        -dt / self._sigma_taus
                    )

                # record value at spike
                means.append(state_mu.sum())
                sigmas.append(state_sigma.sum())

            # Apply nonlinear readout
            means = self.nlin(np.array(means) + self.mu_baseline) * self.mu_scale
            sigmas = (
                self.nlin(np.array(sigmas) + self.sigma_baseline) * self.sigma_scale
            )

            # Sample from gamma distribution
            efficacies = self._sample(means, sigmas, ntrials)

            return means, sigmas, efficacies

        # Standard evaluation (convolution of spiketrain with kernel)
        else:
            spiketrain = get_stimvec(isivec, **kwargs)

            filtered_spiketrain_mu = self.mu_baseline + _convolve_spiketrain_with_kernel(
                spiketrain, self.mu_kernel)

            filtered_spiketrain_sigma = self.sigma_baseline + _convolve_spiketrain_with_kernel(
                spiketrain, self.sigma_kernel)

            nonlinear_readout_mu = self.nlin(filtered_spiketrain_mu) * self.mu_scale
            nonlinear_readout_sigma = self.nlin(filtered_spiketrain_sigma) * self.sigma_scale
            efficacy_train = nonlinear_readout_mu * spiketrain
            means = efficacy_train[np.where(spiketrain == 1)[0]]
            efficacies = self._sample(nonlinear_readout_mu, nonlinear_readout_sigma, ntrials)

            if return_all:
                return {
                    "filtered_spiketrain_mu": filtered_spiketrain_mu,
                    "filtered_spiketrain_sigma": filtered_spiketrain_sigma,
                    "nonlinear_readout": nonlinear_readout_mu,
                    "nonlinear_readout": nonlinear_readout_sigma,
                    "efficacytrain": efficacy_train,
                    "means": means,
                    "efficacies": efficacies,
                }

            else:
                return means, efficacies

    def reset(self):
        pass

#new version of model to handle Gaussian SD with exponential mu kernels
class easySRP(ExpSRP):
    """
    SRP model in which the mu kernel is parameterized by a set of amplitudes
    and respective exponential decay time constants. Variance is treated as
    Gaussian with fixed SD centered on the predicted mean.

    This implementation of the SRP model is used for statistical inference of
    parameters and can be integrated between spikes for efficient numerical
    implementation.
    """

    def __init__(
        self,
        mu_baseline=0,
        mu_amps=[100, 600, 2000],
        mu_taus=[15, 200, 300],
        SD=1,
        mu_scale=None,
        **kwargs
    ):
        #dummy values for super-class
        sigma_baseline = 0
        sigma_amps = [0.1, 0.1, 0.1]
        sigma_taus = [0.1, 0.1, 0.1]

        #build instance of super-class
        super().__init__(
                mu_baseline,
                mu_amps,
                mu_taus,
                sigma_baseline,
                sigma_amps,
                sigma_taus,
                mu_scale=mu_scale
        )

        #save SD as attribute
        self.SD = SD
        self.rng = np.random.default_rng()
    #update sample method
    def _sample(self, mean, sigma, ntrials):
        """
        Samples `ntrials` response amplitudes from a normal distribution
        given mean and sigma

        return: sampled efficacies
        """

        return self.rng.normal(loc=mean, scale=sigma,
            size=(ntrials, len(np.atleast_1d(mean))))

    #override super-class method
    def run_ISIvec(self, isivec, ntrials=1, fast=True, return_all=False, **kwargs):
        """
        Overrides the `run_ISIvec` method because the SRP model with
        exponential decays can be integrated between spikes,
        therefore speeding up computation in some cases
        (if ISIs are large, i.e. presynaptic spikes are sparse)

        return: efficacies
        """

        # Fast evaluation (integrate between spikes)
        if fast:

            state_mu = np.zeros(self._nexp_mu)  # assume kernels have decayed to zero
            means = []

            for spike, dt in enumerate(isivec):

                if spike > 0:
                    # At the first spike, read out baseline efficacy
                    # At every following spike, integrate over the ISI and then read out efficacy
                    state_mu = (state_mu + self._mu_amps) * np.exp(-dt / self._mu_taus)

                # record value at spike
                means.append(state_mu.sum())

            # Apply nonlinear readout
            means = self.nlin(np.array(means) + self.mu_baseline) * self.mu_scale

            # Sample from gamma distribution
            efficacies = self._sample(means, self.SD, ntrials)

            return means, efficacies

        # Standard evaluation (convolution of spiketrain with kernel)
        else:
            spiketrain = get_stimvec(isivec, **kwargs)
            filtered_spiketrain = self.mu_baseline + _convolve_spiketrain_with_kernel(
                spiketrain, self.mu_kernel
            )
            nonlinear_readout = self.nlin(filtered_spiketrain) * self.mu_scale
            efficacy_train = nonlinear_readout * spiketrain
            means = efficacy_train[np.where(spiketrain == 1)[0]]
            efficacies = self._sample(means, self.SD, ntrials)

            if return_all:
                return {
                    "filtered_spiketrain": filtered_spiketrain,
                    "nonlinear_readout": nonlinear_readout,
                    "efficacytrain": efficacy_train,
                    "means": means,
                    "efficacies": efficacies,
                }

            else:
                return means, efficacies

