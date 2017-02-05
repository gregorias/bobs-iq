#!/usr/bin/env python3
# Copyright 2017 Grzegorz Milka grzegorzmilka@gmail.com
#
# An adaptation of the heatmap and density function graph from the
# "Bayesian Benefits for the Pragmatic Researcher" paper.
#
# We symbolize the variable indicating the test results with T, the Bob's IQ
# with U, and test's deviation with E.
#
# The distribution of priors, (U, E), is represented as a 2-dimensional table
# with probability density samples over a domain area (see global constants).
# The values are sometimes not exact but proportional to the density or even
# logarithmized.
# By taking logarithm of these values it is easy to calculate and represent
# P(U, E | T=t), since it is equal to P(T=t | U, E) * P(U, E) / P(T=t)
# P(T=t) is the same over the entire prior domain, so we don't have to
# calculate it and the nominator turns into a sum
# P(T_1=t_1 | U, E) + ... + P(T_n=t_n | U, E) + P(U, E)
#
# The original paper used adaptive rejection metropolis sampling on P(U, E | T)
# to calculate posterior distribution of U (P(U | T)). This is an overkill as
# P(U | T) = \int_E P(U, E | T). So we only need to sum values over the E axis.
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

# Experiment relevant constants
IQ_PRIOR_MU = 75
IQ_PRIOR_SD = 12
SD_PRIOR_LO = 5
SD_PRIOR_HI = 15

IQ_INSTANCES = [
    73,
    67,
    79,
]

# Considered domain ranges of prior values
IQ_LO = 40
IQ_HI = 110
SD_LO = 5
SD_HI = 15

# Number of samples in an univariate distribution domain
DIST_DOMAIN_SIZE = 100


def prior_domain(n):
    """Creates a 2-dimensional array with the censored domain of (U, E)
    Returns:
        2 dimensional numpy (n, n) array with domain values of (U, E).
        The array's dtype is [('u', float), ('e', float)]. Values on both axes
        are uniformly increasing sequences between two constants.
    """
    domain = np.zeros((n, n), dtype=[('u', float), ('e', float)])
    for ((i, u), (j, e)) in itertools.product(
            enumerate(np.linspace(IQ_LO, IQ_HI, n, endpoint=True)),
            enumerate(np.linspace(SD_LO, SD_HI, n, endpoint=True))):
        domain[i, j] = (u, e)
    return domain


def calc_prior_prob(u_e_pair):
    """Calculates the prior probability of (U, E)

    Returns:
        The probability P(U=u, E=e)
    """
    u, e = u_e_pair
    u_prob = stats.norm.pdf(u, IQ_PRIOR_MU, IQ_PRIOR_SD)
    e_prob = stats.uniform.pdf(e, SD_PRIOR_LO, SD_PRIOR_HI - SD_PRIOR_LO)
    return u_prob * e_prob


def calc_joint_prior_dist(u_e_domain):
    """Calculates joint distribution of P(U, E) over u_e_domain."""
    f = np.vectorize(calc_prior_prob)
    return f(u_e_domain)


def calc_iq_prob(iq, u_e_pair):
    """Calculates the probability P(T=iq|U, E) over provided (U, E) domain."""
    u, e = u_e_pair
    return stats.norm.pdf(iq, u, e)


def display_plot(priors_posteriors, iq_posteriors):
    """Displays a grid of plots for provided distributions.

    Args:
        priors_posteriors: a list of 2-dimensional arrays representing a
            posterior distribution of prior parameters. Need not be normalized.
            The probability values should be logarithmized.
        iq_posteriors: a list of 1-dimnesional arrays representing the
            posterior distribution of Bob's IQ.
    """
    assert (DIST_DOMAIN_SIZE == 100)
    assert (priors_posteriors[0].shape == (DIST_DOMAIN_SIZE, DIST_DOMAIN_SIZE))
    assert (iq_posteriors[0].shape == (DIST_DOMAIN_SIZE, ))
    ue = prior_domain(DIST_DOMAIN_SIZE)
    f = plt.figure()
    sns.set_style('whitegrid')
    AXIS_TICK_COUNT = 10
    for i, post in enumerate(priors_posteriors):
        ax = f.add_subplot(2, len(priors_posteriors), i + 1)
        post = np.exp(post + post.min())
        sns.heatmap(
            post.T[::-1, :], cmap='viridis', ax=ax, robust=True, cbar=False)
        ax.xaxis.set_ticks(np.arange(0, DIST_DOMAIN_SIZE + 1, AXIS_TICK_COUNT))
        ax.xaxis.set_ticklabels([
            '{0:0.0f}'.format(iq)
            for iq in np.linspace(IQ_LO, IQ_HI, AXIS_TICK_COUNT + 1)
        ])
        ax.yaxis.set_ticks(np.arange(0, DIST_DOMAIN_SIZE + 1, AXIS_TICK_COUNT))
        ax.yaxis.set_ticklabels([
            '{0:d}'.format(int(dev))
            for dev in np.linspace(SD_PRIOR_LO, SD_PRIOR_HI, AXIS_TICK_COUNT +
                                   1)
        ])
        ax.set_xlabel('IQ posterior probability')
        ax.set_ylabel('Standard deviation posterior probability')

    x = ue['u'][:, 0]
    IQ_DENSITY_YLIM = (0, 0.001)
    for i, iq in enumerate(iq_posteriors):
        ax = f.add_subplot(2,
                           len(iq_posteriors), len(priors_posteriors) + i + 1)
        ax.set_ylim(IQ_DENSITY_YLIM)
        plotline = ax.plot(x, iq)[0]
        if i > 0:
            ax.plot(
                x, iq_posteriors[i - 1], color=plotline.get_color(), alpha=0.5)
        ax.set_xlabel('IQ posterior probability')
        ax.set_ylabel('Density')

    plt.show()


def calculate_all_distributions():
    """
    Returns:
        A tuple (priors_posteriors, iq_posteriors) which are the arguments
        required by 'display_plot'.
    """
    priors_posteriors = []
    iq_posteriors = []

    u_e_domain = prior_domain(DIST_DOMAIN_SIZE)
    priors_posteriors.append(np.log(calc_joint_prior_dist(u_e_domain)))
    for iq in IQ_INSTANCES:
        f = np.vectorize(lambda u_e_pair: calc_iq_prob(iq, u_e_pair))
        likelihood = np.log(f(u_e_domain))
        priors_posteriors.append(priors_posteriors[-1] + likelihood)

    for pp in priors_posteriors:
        pp += pp.max()
        pp = np.exp(pp)
        pp = pp / np.sum(pp)
        iq_density_approx = pp.sum(axis=1)
        iq_posterior = iq_density_approx / (IQ_HI - IQ_LO)
        iq_posteriors.append(iq_posterior)

    return priors_posteriors, iq_posteriors


if __name__ == '__main__':
    prior, iq = calculate_all_distributions()
    display_plot(prior, iq)
