
import pymc as pm
import numpy as np
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns

class BVAR:
    def __init__(self, endog, exog, lags, endog_names=None, exog_names=None):
        """
        Bayesian Vector Autoregression (BVAR) model.

        Parameters:
        -----------
        endog : array-like (n_periods, n_endog)
            Endogenous variables (e.g., base sales, brand awareness, consideration)
        exog : array-like (n_periods, n_exog)
            Exogenous variables (e.g., marketing spend by channel)
        lags : int
            Number of lags to include in the VAR model
        endog_names : list of str, optional
            Names of endogenous variables
        exog_names : list of str, optional
            Names of exogenous variables
        """
        self.endog = np.array(endog)
        self.exog = np.array(exog)
        self.lags = lags
        self.model = None
        self.trace = None

        # Set default names if not provided
        self.endog_names = endog_names or [f"endog_{i}" for i in range(self.endog.shape[1])]
        self.exog_names = exog_names or [f"exog_{i}" for i in range(self.exog.shape[1])]

        # Prepare lagged data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare lagged endogenous variables for VAR model."""
        n_periods, n_endog = self.endog.shape

        # Create lagged endogenous matrix
        # For each period t, include Y[t-1], Y[t-2], ..., Y[t-L]
        lagged_endog_list = []
        for lag in range(1, self.lags + 1):
            lagged = np.roll(self.endog, lag, axis=0)
            lagged[:lag, :] = 0  # Set initial periods to 0
            lagged_endog_list.append(lagged)

        # Stack: [Y_t-1, Y_t-2, ..., Y_t-L]
        self.lagged_endog = np.concatenate(lagged_endog_list, axis=1)

        # Remove first 'lags' periods (no valid history)
        self.Y = self.endog[self.lags:]
        self.X_lagged = self.lagged_endog[self.lags:]
        self.X_exog = self.exog[self.lags:]

        self.n_obs = len(self.Y)

    def build_model(self):
        """Build BVAR model with VAR structure and exogenous variables."""
        n_endog = self.endog.shape[1]
        n_exog = self.exog.shape[1]
        n_lagged = n_endog * self.lags

        with pm.Model() as self.model:
            # Priors for the VAR coefficients (A)
            # Shape: (n_endog, n_endog * lags)
            # A[i, j] is the effect of lagged variable j on current variable i
            A = pm.Normal('A', mu=0, sigma=1, shape=(n_endog, n_lagged))

            # Priors for the exogenous coefficients (B)
            # Shape: (n_endog, n_exog)
            # B[i, j] is the effect of exogenous variable j on endogenous variable i
            B = pm.Normal('B', mu=0, sigma=1, shape=(n_endog, n_exog))

            # Priors for the covariance matrix using LKJ Cholesky
            chol, corr, stds = pm.LKJCholeskyCov(
                'chol',
                n=n_endog,
                eta=2.0,
                sd_dist=pm.HalfNormal.dist(sigma=1.0),
                compute_corr=True
            )

            # VAR model: Y_t = A @ Y_{t-1:t-L} + B @ X_t + epsilon
            # Compute predicted values for each observation
            mu = pm.Deterministic(
                'mu',
                pm.math.dot(self.X_lagged, A.T) + pm.math.dot(self.X_exog, B.T)
            )

            # Likelihood: Multivariate Normal
            # Y[t] ~ MVN(mu[t], Sigma)
            Y_obs = pm.MvNormal(
                'Y_obs',
                mu=mu,
                chol=chol,
                observed=self.Y
            )

    def fit(self, draws=1000, tune=1000):
        """
        Fit the BVAR model using MCMC sampling.

        Parameters:
        -----------
        draws : int
            Number of MCMC samples to draw
        tune : int
            Number of tuning/warmup samples
        """
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, cores=1, random_seed=42)

    def calculate_irf(self, periods=24, shock_size=1.0):
        """
        Calculate Impulse Response Functions (IRF).

        Traces the dynamic response of endogenous variables to a one-time shock
        in exogenous variables (marketing spend).

        Parameters:
        -----------
        periods : int
            Number of periods to simulate forward (default: 24 weeks)
        shock_size : float
            Size of the shock to apply (default: 1.0 = $1M or 1 unit)

        Returns:
        --------
        irf_results : dict
            Dictionary containing IRF arrays for each exog → endog relationship
            Keys: "exog_{i}_to_endog_{j}"
            Values: array of shape (periods,) showing response over time
        """
        if self.trace is None:
            raise ValueError("Model must be fit before calculating IRF")

        # Get posterior means of coefficients
        A_mean = self.trace.posterior['A'].mean(dim=['chain', 'draw']).values
        B_mean = self.trace.posterior['B'].mean(dim=['chain', 'draw']).values

        n_endog = self.endog.shape[1]
        n_exog = self.exog.shape[1]

        # Initialize IRF storage
        irf_results = {}

        # For each exogenous variable (marketing channel)
        for exog_idx in range(n_exog):
            # Create shock vector: 1 unit shock to channel exog_idx at time 0
            shock = np.zeros(n_exog)
            shock[exog_idx] = shock_size

            # Simulate forward for each endogenous variable
            for endog_idx in range(n_endog):
                # Initialize response trajectory
                response = np.zeros(periods)

                # Initialize state: Y[t-L], ..., Y[t-1]
                state = np.zeros((self.lags, n_endog))

                # Simulate forward
                for t in range(periods):
                    # Flatten lagged state: [Y[t-1], Y[t-2], ..., Y[t-L]]
                    lagged_flat = state[::-1].flatten()  # Reverse order for most recent first

                    # Compute prediction: A @ lagged + B @ shock
                    if t == 0:
                        # Apply shock at t=0
                        pred = A_mean @ lagged_flat + B_mean @ shock
                    else:
                        # No shock after t=0
                        pred = A_mean @ lagged_flat

                    # Store response for this endogenous variable
                    response[t] = pred[endog_idx]

                    # Update state: shift and add new prediction
                    state = np.roll(state, 1, axis=0)
                    state[0] = pred

                # Store IRF
                key = f"{self.exog_names[exog_idx]}_to_{self.endog_names[endog_idx]}"
                irf_results[key] = response

        self.irf_results = irf_results
        return irf_results

    def plot_irf(self, irf=None, figsize=(15, 10)):
        """
        Plot Impulse Response Functions.

        Parameters:
        -----------
        irf : dict, optional
            IRF results from calculate_irf(). If None, uses self.irf_results
        figsize : tuple
            Figure size (width, height)
        """
        if irf is None:
            if not hasattr(self, 'irf_results'):
                raise ValueError("No IRF results found. Run calculate_irf() first.")
            irf = self.irf_results

        n_exog = len(self.exog_names)
        n_endog = len(self.endog_names)

        # Create subplot grid
        fig, axes = plt.subplots(n_endog, n_exog, figsize=figsize, sharex=True)

        if n_endog == 1:
            axes = axes.reshape(1, -1)
        if n_exog == 1:
            axes = axes.reshape(-1, 1)

        # Plot each IRF
        for exog_idx, exog_name in enumerate(self.exog_names):
            for endog_idx, endog_name in enumerate(self.endog_names):
                ax = axes[endog_idx, exog_idx]

                key = f"{exog_name}_to_{endog_name}"
                if key in irf:
                    response = irf[key]
                    periods = len(response)
                    time_idx = np.arange(periods)

                    # Plot response
                    ax.plot(time_idx, response, linewidth=2, color='steelblue')
                    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                    ax.fill_between(time_idx, 0, response, alpha=0.3, color='steelblue')

                    # Labels
                    ax.set_title(f"{exog_name} → {endog_name}", fontsize=10, fontweight='bold')
                    ax.set_ylabel('Response', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # Only show x-label on bottom row
                    if endog_idx == n_endog - 1:
                        ax.set_xlabel('Periods', fontsize=9)

        plt.tight_layout()
        return fig

    def calculate_long_term_roi(self, irf=None, sales_var_name=None):
        """
        Calculate long-term ROI from brand-building effects.

        Integrates the IRF to estimate cumulative sales lift from a $1 marketing shock.

        Parameters:
        -----------
        irf : dict, optional
            IRF results. If None, uses self.irf_results
        sales_var_name : str, optional
            Name of the sales endogenous variable (default: first endog variable)

        Returns:
        --------
        roi_dict : dict
            Long-term ROI per marketing channel
            Keys: channel names
            Values: cumulative sales lift per $1 spent
        """
        if irf is None:
            if not hasattr(self, 'irf_results'):
                raise ValueError("No IRF results found. Run calculate_irf() first.")
            irf = self.irf_results

        # Determine which endogenous variable represents sales
        if sales_var_name is None:
            sales_var_name = self.endog_names[0]  # Default to first variable
            print(f"Assuming '{sales_var_name}' represents sales variable")

        roi_dict = {}

        # For each marketing channel (exogenous variable)
        for exog_name in self.exog_names:
            key = f"{exog_name}_to_{sales_var_name}"

            if key in irf:
                # Cumulative sales lift = sum of IRF over all periods
                cumulative_lift = np.sum(irf[key])

                # ROI = cumulative_lift / $1 shock
                roi_dict[exog_name] = cumulative_lift
            else:
                print(f"Warning: IRF key '{key}' not found. Skipping {exog_name}")

        return roi_dict

    def plot_long_term_roi(self, roi_dict):
        """
        Visualize long-term ROI by channel.

        Parameters:
        -----------
        roi_dict : dict
            ROI per channel from calculate_long_term_roi()
        """
        channels = list(roi_dict.keys())
        roi_values = list(roi_dict.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar chart
        colors = sns.color_palette("viridis", len(channels))
        bars = ax.bar(channels, roi_values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'${height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        ax.set_ylabel('Long-Term ROI ($ per $ spent)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Marketing Channel', fontsize=12, fontweight='bold')
        ax.set_title('Long-Term ROI from Brand-Building Effects', fontsize=14, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def summary(self):
        """Return summary statistics of the posterior distribution."""
        if self.trace is None:
            raise ValueError("Model must be fit before getting summary")
        return pm.summary(self.trace)
