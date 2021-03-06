"""Methods for the `Particles` objects to deal with survival analysis."""

from __future__ import absolute_import, division, print_function
import lifelines
import numpy as np
import pandas as pd

@property
def dwell_time(self):
    """The number of frames this particle was observed."""
    return self['frame'].max() - self['frame'].min()


def half_life(self, alpha=0.05):
    """Calculate the half-life of the particle.

    Parameters:
    -----------
    alpha: float
        The significance level for the interval of confidence.

    Returns:
    --------
    coefficients: np.array
        The half-life of the exponential decay as well as it lower and upper bound at given alpha.

    Notes:
    -------
    Half-life is sqrt(2) / survival coefficient (lambda)
    For now, all survival distributions are fitted as exponential decays. Look into the [`lifelines` library](http://lifelines.readthedocs.io/en/latest/) for more distribution and information on survival analysis.


    """
    lambda_ = self.survival_coefficient(alpha)
    return np.log(2) / lambda_


def lifetime(self, alpha=0.05):
    """Calculate the lifetime of the particle.

    Parameters:
    -----------
    alpha: float
        The significance level for the interval of confidence.

    Returns:
    --------
    coefficients: np.array
        The lifetime coefficient (tau) of the exponential decay as well as it lower and upper bound at given alpha.

    Notes:
    -------
    Lifetime is 1 / survival coefficient (lambda)
    For now, all survival distributions are fitted as exponential decays. Look into the [`lifelines` library](http://lifelines.readthedocs.io/en/latest/) for more distribution and information on survival analysis.


    """
    lambda_ = self.survival_coefficient(alpha)
    return 1 / lambda_


def survival_coefficient(self, alpha=0.05):
    """Calculate the lifetime of the particle.

    Parameters:
    -----------
    alpha: float
        The significance level for the interval of confidence.

    Returns:
    --------
    coefficients: np.array
        The survival coefficient (lambda) of the exponential decay as well as it lower and upper bound at given alpha.

    Notes:
    -------
    For now, all survival distributions are fitted as exponential decays. Look into the [`lifelines` library](http://lifelines.readthedocs.io/en/latest/) for more distribution and information on survival analysis.

    """
    sd = self.survival_distribution()
    sd.alpha = 1 - alpha  # There an error in lifelines. They call the confidence level 'alpha'.
    values = sd.summary[['coef', 'lower {0}'.format(round(sd.alpha, 2)), 'upper {0}'.format(round(sd.alpha, 2))]].values[0]
    return values


def survival_distribution(self):
    """Return the survival distribution of the particles.

    Returns
    --------
    distribution: lifelines.ExponentialFitter
        The distribution of the particle's dwell times.

    Notes:
    -------
    For now, all survival distributions are fitted as exponential decays. Look into the [`lifelines` library](http://lifelines.readthedocs.io/en/latest/) for more distribution and information on survival analysis.

    """
    sd = lifelines.ExponentialFitter()
    sd.fit([particle.dwell_time for particle in self], event_observed=[particle['frame'].max() + particle.tracking_parameters['memory'] < particle.source.shape[0] for particle in self])
    return sd


def survival_events(self):
    """Return a data frame of the survival events that will be used for making distributions.
    
    Returns
    -------
    events: pd.DataFrame
        The events dwell times and whether they ended within the observation window.
    """
    return pd.DataFrame([(particle.dwell_time, particle['frame'].max() + particle.tracking_parameters['memory'] < particle.source.shape[0]) for particle in self], columns=['dwell_time', 'end_observed'])

_extension = {
    'Particle': {
        'dwell_time': dwell_time
    },
    'Particles': {
        'survival_distribution': survival_distribution,
        'survival_coefficient': survival_coefficient,
        'lifetime': lifetime,
        'half_life': half_life,
        'survival_events': survival_events
    }
}
