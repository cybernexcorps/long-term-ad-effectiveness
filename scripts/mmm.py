
import pymc as pm
import numpy as np
import pytensor.tensor as pt

class UCM_MMM:
    def __init__(self, sales_data, marketing_data, max_lag=4):
        """
        Initialize UCM-MMM model.

        Parameters:
        -----------
        sales_data : array-like
            Sales/revenue time series
        marketing_data : array-like (n_periods, n_channels)
            Marketing spend by channel
        max_lag : int
            Maximum lag for adstock transformation (default: 4 weeks)
        """
        self.sales_data = sales_data
        self.marketing_data = marketing_data
        self.max_lag = max_lag
        self.model = None
        self.trace = None

    def build_model(self):
        """Build UCM-MMM model with adstock and saturation transformations."""
        with pm.Model() as self.model:
            # Priors for adstock decay rates (one per channel)
            alpha = pm.Beta("alpha", 2.0, 2.0, shape=self.marketing_data.shape[1])

            # Adstock transformation
            adstocked_marketing = pm.Deterministic(
                "adstocked_marketing",
                self._adstock(self.marketing_data, alpha)
            )

            # Priors for saturation parameters
            # Lambda: half-saturation point (spend level where response is 50% of max)
            lam = pm.Gamma("lambda", alpha=2.0, beta=0.0001, shape=self.marketing_data.shape[1])

            # Kappa: shape parameter (steepness of the curve)
            kappa = pm.Gamma("kappa", alpha=2.0, beta=1.0, shape=self.marketing_data.shape[1])

            # Apply saturation transformation
            saturated_marketing = pm.Deterministic(
                "saturated_marketing",
                self._saturation(adstocked_marketing, lam, kappa)
            )

            # Priors for marketing effect (elasticity per channel)
            beta = pm.HalfNormal("beta", 1.0, shape=self.marketing_data.shape[1])

            # Total marketing contribution
            marketing_effect = pm.Deterministic(
                "marketing_effect",
                pm.math.dot(saturated_marketing, beta)
            )

            # Baseline (intercept + trend)
            baseline = pm.Normal("baseline", mu=self.sales_data.mean(), sigma=self.sales_data.std())

            # Optional: Add time trend
            n_periods = len(self.sales_data)
            time_idx = np.arange(n_periods)
            trend_coef = pm.Normal("trend", mu=0, sigma=1000)
            trend = pm.Deterministic("trend_effect", trend_coef * time_idx)

            # Expected sales
            mu = baseline + marketing_effect + trend

            # Likelihood
            sigma = pm.HalfNormal("sigma", sigma=self.sales_data.std())
            y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=self.sales_data)

    def _adstock(self, x, alpha):
        """
        Geometric adstock transformation with carryover effects.

        The effect at time t includes contributions from current and past periods:
        y[t] = x[t] + alpha * x[t-1] + alpha^2 * x[t-2] + ... + alpha^L * x[t-L]

        Parameters:
        -----------
        x : array (n_periods, n_channels)
            Marketing spend by channel
        alpha : array (n_channels,)
            Decay rate per channel (0 = no carryover, 1 = infinite carryover)

        Returns:
        --------
        adstocked : array (n_periods, n_channels)
            Adstocked marketing spend
        """
        n_periods, n_channels = x.shape

        # Create weight matrix for geometric adstock
        # weights[i, j] = alpha^i for lag i
        lags = np.arange(self.max_lag + 1)  # [0, 1, 2, ..., max_lag]

        # For each channel, compute adstocked values
        adstocked_channels = []

        for channel_idx in range(n_channels):
            # Get spending for this channel
            x_channel = x[:, channel_idx]
            alpha_channel = alpha[channel_idx]

            # Compute weights: [1, alpha, alpha^2, alpha^3, ...]
            weights = pt.power(alpha_channel, lags)

            # Normalize weights to sum to 1 (optional, for interpretability)
            weights = weights / pt.sum(weights)

            # Apply convolution: each period gets weighted sum of past spends
            # For simplicity, use a loop (PyMC will handle autodiff)
            adstocked = []
            for t in range(n_periods):
                # Sum contributions from current and past periods
                contribution = 0.0
                for lag in range(min(t + 1, self.max_lag + 1)):
                    contribution += x_channel[t - lag] * weights[lag]
                adstocked.append(contribution)

            adstocked_channels.append(pt.stack(adstocked))

        # Stack channels back together
        return pt.stack(adstocked_channels, axis=1)

    def _saturation(self, x, lam, kappa):
        """
        Hill saturation transformation for diminishing returns.

        Uses the Hill equation: y = x^kappa / (lambda^kappa + x^kappa)

        Parameters:
        -----------
        x : array (n_periods, n_channels)
            Adstocked marketing spend
        lam : array (n_channels,)
            Half-saturation point per channel
        kappa : array (n_channels,)
            Shape parameter (steepness) per channel

        Returns:
        --------
        saturated : array (n_periods, n_channels)
            Saturated marketing effect (bounded [0, 1])
        """
        # Apply Hill transformation element-wise
        # y = x^kappa / (lambda^kappa + x^kappa)
        x_powered = pt.power(x, kappa)
        lam_powered = pt.power(lam, kappa)

        saturated = x_powered / (lam_powered + x_powered)

        return saturated

    def fit(self, draws=1000, tune=1000):
        """
        Fit the model using MCMC sampling.

        Parameters:
        -----------
        draws : int
            Number of MCMC samples to draw
        tune : int
            Number of tuning/warmup samples
        """
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, cores=1, random_seed=42)

    def summary(self):
        """Return summary statistics of the posterior distribution."""
        return pm.summary(self.trace)

    def get_base_sales(self):
        """
        Extract base sales (trend component) for use in BVAR model.

        Returns:
        --------
        base_sales : array
            Sales minus marketing effects (baseline + trend)
        """
        if self.trace is None:
            raise ValueError("Model must be fit before extracting base sales")

        # Get posterior mean of baseline and trend
        baseline_mean = self.trace.posterior['baseline'].mean().values
        trend_mean = self.trace.posterior['trend_effect'].mean(dim=['chain', 'draw']).values
        marketing_effect_mean = self.trace.posterior['marketing_effect'].mean(dim=['chain', 'draw']).values

        # Base sales = observed sales - marketing effect
        base_sales = self.sales_data - marketing_effect_mean

        return base_sales

    def calculate_short_term_roi(self):
        """
        Calculate short-term ROI per channel.

        Returns:
        --------
        roi_dict : dict
            ROI per channel (revenue per dollar spent)
        """
        if self.trace is None:
            raise ValueError("Model must be fit before calculating ROI")

        # Get posterior mean of beta (elasticity)
        beta_mean = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values

        # Calculate incremental revenue per channel
        # Total effect = beta * saturated_adstocked_spend
        saturated_marketing_mean = self.trace.posterior['saturated_marketing'].mean(dim=['chain', 'draw']).values

        incremental_revenue = beta_mean * saturated_marketing_mean.sum(axis=0)
        total_spend = self.marketing_data.sum(axis=0)

        # ROI = incremental_revenue / total_spend
        roi = incremental_revenue / total_spend

        roi_dict = {f"channel_{i}": roi[i] for i in range(len(roi))}

        return roi_dict
