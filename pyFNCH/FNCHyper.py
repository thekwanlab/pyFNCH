from abc import ABC, abstractmethod
import math
import logging
import numpy as np
from scipy import optimize, stats


class FNCHyper:
    def __init__(
        self, n_success: int, n_failure: int, m_draws: int, weight: float = 1.0
    ):
        """Construct a FNCH distribution object. Notation follows Liao, Rosen 2001

        Args:
            n_success (int): Number of SUCCESS objects in the urn
            n_failure (int): Number of FAIL objects in the urn
            m_draws (int): Number of draws from the urn
            weight (int, optional): The odds ratio of SUCCESS probability compared to FAIL probability. Defaults to 1.
        """
        self.n_success = n_success
        self.n_failure = n_failure
        self.m_draws = m_draws
        self.weight = weight

        self.support_lower = max(0, self.m_draws - n_failure)
        self.support_upper = min(self.m_draws, n_success)
        self.support = [i for i in range(self.support_lower, self.support_upper + 1)]
        self.fast_support, self.pmf_lookup = self.set_probabilities()

    def r(self, z: int):
        """Recurrence relation to help avoid combinatorial explosions

        Args:
            z (int): integer in the support of the distribution

        Returns:
            int: a large integer
        """
        return (
            self.weight
            * (self.n_success - z + 1)
            * (self.m_draws - z + 1)
            / (z * (self.n_failure - self.m_draws + z))
        )

    def set_probabilities(self):
        """Uses the recurrence relations defined in Liao and Rosen to simultaneously compute all values of the
        probability function on the entire support

        Returns:
            tuple:
                list[int] : Item 0 contains the fast support (values where the PMF is greater than some small threshold)
                dict[int, float] : Item 1 contains the PMF function results as a dictionary
        """
        DELTA = 1e-14
        EPSILON = DELTA / 10

        f = {}
        mode = self.mode()
        f[mode] = 1
        fast_support_upper = mode
        fast_support_lower = mode
        for i in range(mode + 1, self.support_upper + 1):
            fast_support_upper = i
            if f[i - 1] < EPSILON and self.r(i) < 5 / 6:
                break
            else:
                f[i] = f[i - 1] * self.r(i)
        for i in range(mode - 1, self.support_lower - 1, -1):
            fast_support_lower = i
            if f[i + 1] < EPSILON and self.r(i + 1) > 6 / 5:
                break
            else:
                f[i] = f[i + 1] / self.r(i + 1)

        f_sum = sum(f.values())
        fast_support = [i for i in range(fast_support_lower, fast_support_upper + 1)]
        pmf_lookup = {x: f[x] / f_sum for x in f}
        return fast_support, pmf_lookup

    def pmf(self, x: int):
        """Probability Mass Function of the distribution

        Args:
            x (int): number in the support of the distribution

        Raises:
            TypeError: If x is not an int

        Returns:
            float: probability
        """
        if not isinstance(x, int):
            raise TypeError("FNCH is a discrete distribution. Enter an integer X")
        return self.pmf_lookup.get(x, 0)

    def mean(self):
        """Mean of the distribution over the fast support

        Returns:
            float: expected value
        """
        if not hasattr(self, "fast_support"):
            self.set_probabilities()
        return sum([i * self.pmf(i) for i in self.fast_support])

    def mean_fast(self):
        """Mean of the distribution computed using Liao and Rosen approximation

        Returns:
            float: expected value
        """
        A = self.weight - 1
        B = (
            self.m_draws
            - self.n_failure
            - self.weight * (self.n_success + self.m_draws)
        )
        C = self.weight * self.m_draws * self.n_success
        return -2 * C / (B - math.sqrt(B**2 - 4 * A * C))

    def variance(self):
        """Variance fo the distribution over the fast support

        Returns:
            float: variance
        """
        if not hasattr(self, "fast_support"):
            self.set_probabilities()
        mean = self.mean()
        return sum([i**2 * self.pmf(i) for i in self.fast_support]) - mean**2

    def variance_fast(self):
        """Variance of the distribution over using Liao and Rosen approximation

        Returns:
            float: variance
        """
        mean = self.mean_fast()
        N = self.n_success + self.n_failure
        return (N / (N - 1)) / (
            1 / mean
            + 1 / (self.n_success - mean)
            + 1 / (self.m_draws - mean)
            + 1 / (mean + self.n_failure - self.m_draws)
        )

    def mode(self):
        """Mode (highest probability value) of the distribution over the full support

        Returns:
            _type_: _description_
        """
        if self.weight == 1:
            return math.floor(
                ((self.n_success + 1) * (self.m_draws + 1))
                / (self.n_success + self.n_failure + 2)
            )

        r_results = [self.r(z) for z in self.support[1:]]
        if max(r_results) <= 1:
            return self.support_lower
        if min(r_results) >= 1:
            return self.support_upper

        A = self.weight - 1
        B = -1 * (
            (self.n_success + self.m_draws + 2) * self.weight
            + self.n_failure
            - self.m_draws
        )
        C = self.weight * (self.n_success + 1) * (self.m_draws + 1)
        q = -0.5 * (B + (1 if B >= 0 else -1) * math.sqrt(B**2 - 4 * A * C))
        mode = int(C / q)
        return mode


class MLE(ABC):
    @abstractmethod
    def __init__(
        self,
        n_success: int,
        n_failure: int,
        m_draws: int,
        significance_threshold: float = 0.05,
    ):
        """Facilitate maximum likelihood estimation for the weight parameter of a FNCH distribution

        Args:
            n_success (int): number of SUCCESS balls in the urn
            n_failure (int): number of FAILURE bals in the urn
            m_draws (int): number of balls drawn from the urn
            significance_threshold (float, optional): Threshold used to compute confidence intervals. Defaults to 0.05.
        """
        self.n_success = n_success
        self.n_failure = n_failure
        self.m_draws = m_draws
        self.alpha = significance_threshold

    def factory(self, weight: float):
        """A FNCH object factory, used internally by estimation functions

        Args:
            weight (float): weight parameter of an FNCH distribution

        Returns:
            FNCHyper
        """
        return FNCHyper(
            n_success=self.n_success,
            n_failure=self.n_failure,
            m_draws=self.m_draws,
            weight=weight,
        )

    def likelihood(self, weight: float, x: int):
        """Likelihood function of an observation from Maximum Likelihood Estimation

        Args:
            weight (float): weight parameter of an FNCH distribution
            x (float): observation from an FNCH distribution

        Returns:
            float: probability
        """
        return self.factory(weight).pmf(x)

    @abstractmethod
    def score(self, weight: float, x: int):
        """Score function of an observation from Maximum Likelihood Estimation

        Args:
            weight (float): weight parameter of an FNCH distribution
            x (int): observation from an FNCH distribution

        Returns:
            float: score value
        """
        pass

    @abstractmethod
    def information(self, weight: float, x: int):
        """Information function of an observation from Maximum Likelihood Estimation

        Args:
            weight (float): weight parameter of an FNCH distribution
            x (int): observation from an FNCH distribution

        Returns:
            float: information value
        """
        pass

    @abstractmethod
    def fisher_information(self, weight: float):
        """Fisher Information of an observation from Maximum Likelihood Estimation
        This is defined as the expected value of the Information

        Args:
            weight (float): weight parameter of an FNCH distribution

        Returns:
            float: Fisher information value
        """
        pass

    def likehood_ratio_statistic(self, weight: float, null_weight: float, x: int):
        """Likelihood Ratio test statistic for weight paramters considering the observed data, compared to an Null Hypothesis weight

        Args:
            weight (float): weight parameter
            null_weight (float): weight parameter under the null hypothesis
            x (int): observed value

        Returns:
            float: LR statistic, distributed as Chi-square with 1 degree of freedom
        """
        return -2 * math.log(
            1e-14 + self.likelihood(weight, x) / self.likelihood(null_weight, x)
        )

    def likelihood_ratio_minus_ci_bound(
        self, weight: float, null_weight: float, x: int
    ):
        """Likelihood Ratio test statistic for weight paramters considering the observed data, compared to an Null Hypothesis weight
        then subtracts a threshold depending on the chosen
        significance level for purposes of confidence interval construction

        Args:
            weight (float): weight parameter
            null_weight (float): weight parameter under the null hypothesis
            x (int): observed value

        Returns:
            float: LR statistic minus threshold. Values below 0 represent statistical significant results under the null
        """
        return self.likehood_ratio_statistic(weight, null_weight, x) - stats.chi2.isf(
            self.alpha, 1
        )

    def score_statistic(self, weight: float, x: int):
        """Score test statistic for weight paramters considering the observed data

        Args:
            weight (float): weight parameter
            x (int): observed value

        Returns:
            float: Score statistic, distributed as Chi-square with 1 degree of freedom
        """
        return self.score(weight, x) ** 2 / self.fisher_information(weight)

    def score_statistic_minus_ci_bound(self, weight, x):
        """Score test statistic for weight paramters considering the observed data
        then subtracts a threshold depending on the chosen
        significance level for purposes of confidence interval construction

        Args:
            weight (float): weight parameter
            x (int): observed value

        Returns:
            float: Score statistic minus threshold. Values below 0 represent statistical significant results under the null
        """
        return self.score_statistic(weight, x) - stats.chi2.isf(self.alpha, 1)

    def fit(self, x: int):
        search_lower_bound, search_upper_bound = 1e-5, 1e5

        try:
            # this error handling is very reusable, for the mle and the confidence interval bounds
            # but it needs refactoring to be put to use in a better way - I got lazy
            maximum_likelihood_estimate = optimize.brentq(
                self.score, a=search_lower_bound, b=search_upper_bound, args=(x)
            )
        except ValueError as e:
            logging.warning(
                f"The maximum likelihood estimate might be out of our search bounds (f{search_lower_bound} to {search_upper_bound}). Returning 0 or np.inf"
            )
            logging.warning(e)
            central_hypergeometric_mode = self.factory(weight=1).mode()
            if central_hypergeometric_mode < x:
                return MaximumLikelihoodEstimate(mle=np.inf, alpha=self.alpha)
            else:
                return MaximumLikelihoodEstimate(mle=0.0, alpha=self.alpha)

        try:
            # Likelihood Ratio confidence interval
            lr_lower = optimize.brentq(
                self.likelihood_ratio_minus_ci_bound,
                a=search_lower_bound,
                b=maximum_likelihood_estimate,
                args=(maximum_likelihood_estimate, x),
            )
            lr_upper = optimize.brentq(
                self.likelihood_ratio_minus_ci_bound,
                a=maximum_likelihood_estimate,
                b=search_upper_bound,
                args=(maximum_likelihood_estimate, x),
            )
            lr_ci = (lr_lower, lr_upper)

            # Score statistic confidence interval
            score_lower = optimize.brentq(
                self.score_statistic_minus_ci_bound,
                a=search_lower_bound,
                b=maximum_likelihood_estimate,
                args=(x),
            )
            score_upper = optimize.brentq(
                self.score_statistic_minus_ci_bound,
                a=maximum_likelihood_estimate,
                b=search_upper_bound,
                args=(x),
            )
            score_ci = (score_lower, score_upper)

            # Wald confidence interval
            wald_lower = maximum_likelihood_estimate - stats.norm.isf(
                self.alpha / 2
            ) / math.sqrt(self.fisher_information(maximum_likelihood_estimate))
            wald_upper = maximum_likelihood_estimate + stats.norm.isf(
                self.alpha / 2
            ) / math.sqrt(self.fisher_information(maximum_likelihood_estimate))
            wald_ci = (wald_lower, wald_upper)

        except ValueError as e:
            logging.warning(
                f"A confidence interval bound might be out of our search bounds (f{search_lower_bound} to {search_upper_bound}). Not returning any confidence intervals"
            )
            logging.warning(e)
            return MaximumLikelihoodEstimate(mle=maximum_likelihood_estimate, alpha=self.alpha)

        return MaximumLikelihoodEstimate(
            mle=maximum_likelihood_estimate,
            alpha=self.alpha,
            lr_ci_bounds=lr_ci,
            score_ci_bounds=score_ci,
            wald_ci_bounds=wald_ci,
        )

    def odds_to_probability(odds):
        """Convert odds of success to probability of success"""
        return odds / (odds + 1)

    def round2(x):
        """Round a number to 2 decimal places"""
        return round(x, 2)


class ExactEstimation(MLE):
    def __init__(
        self,
        n_success: int,
        n_failure: int,
        m_draws: int,
        significance_threshold: float = 0.05,
    ):
        super().__init__(n_success, n_failure, m_draws, significance_threshold)

    def score(self, weight: float, x: int):
        return (x - self.factory(weight).mean()) / weight

    def information(self, weight, x):
        fnch = self.factory(weight)
        mean = fnch.mean()
        var = fnch.variance()
        return (1 / weight**2) * (x + var - mean)

    def fisher_information(self, weight):
        return self.factory(weight).variance() / weight**2


class FastEstimation(MLE):
    def __init__(
        self,
        n_success: int,
        n_failure: int,
        m_draws: int,
        significance_threshold: float = 0.05,
    ):
        super().__init__(n_success, n_failure, m_draws, significance_threshold)

    def score(self, weight, x):
        return (x - self.factory(weight).mean_fast()) / weight

    def information(self, weight, x):
        fnch = self.factory(weight)
        mean = fnch.mean_fast()
        var = fnch.variance_fast()
        return (1 / weight**2) * (x + var - mean)

    def fisher_information(self, weight):
        return self.factory(weight).variance_fast() / weight**2


class MaximumLikelihoodEstimate:
    """Contains the important bits of a Maximum Likelihood Estimate for any parameter.

    Attributes:

        _mle: the maximum likelihood estimate
        _alpha: the significance threshold used
        _lr_ci: the confidence interval computed from the likelihood ratio test statistic
        _score_ci: the confidence interval computed from the score statistic
        _wald_ci: the Wald confidence interval
    """

    def __init__(
        self,
        mle: float,
        alpha: float,
        lr_ci_bounds: tuple[float, float] = None,
        score_ci_bounds: tuple[float, float] = None,
        wald_ci_bounds: tuple[float, float] = None,
    ):
        self._mle = mle
        self._alpha = alpha
        self._lr_ci = lr_ci_bounds
        self._score_ci = score_ci_bounds
        self._wald_ci = wald_ci_bounds

    def __repr__(self):
        return f"""<MaximumLikelihoodEstimate> {self._mle:0.4f}
Confidence intervals computed at significance threshold {self._alpha:0.2f}
    Likelihood Ratio : ({self._lr_ci[0]:0.2f}, {self._lr_ci[1]:0.2f})
    Score            : ({self._score_ci[0]:0.2f}, {self._score_ci[1]:0.2f})
    Wald             : ({self._wald_ci[0]:0.2f}, {self._wald_ci[1]:0.2f})
        """

    def __str__(self):
        return f"Maximum Likelihood Estimate: {self._mle:0.4}"


pass
