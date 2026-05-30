"""Build inflation_forecast_comparison_v2.ipynb with six improvements:
   1. Multi-seed DNN averaging (5 seeds, mean prediction)
   2. Huber loss (delta=1.0) instead of MSE
   3. First-difference money_growth before scaling
   4. Rolling 12-month inflation volatility as covariate (shift(1) to avoid look-ahead)
   5. 50/50 train/test split (180/180) with expanding-window forecasts
   6. Bootstrap pre-training on 20 synthetic DGP replications
"""
import json
import nbformat as nbf

OUT_PATH = r"G:\My Drive\Colab Notebooks\ML_WAIFEM_2026\ML\inflation_forecast_comparison_v2.ipynb"

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ─────────────────────────────────────────────────────────────────────────────
md("""# Inflation Forecast Horse Race — v2
**Six methodological upgrades over v1** (all targeted at giving DNNs a fair shot):

1. **Multi-seed averaging** — 5 seeds per DNN, predictions are the *mean* across seeds
2. **Huber loss (δ=1.0)** — robust to jumps and heavy-tailed shocks
3. **First-differenced `money_growth`** — removes the unit root before scaling
4. **Rolling 12-month inflation volatility** as an additional input (shift(1), no look-ahead) — gives DNNs the SV signal that Block-2 SV models get for free
5. **50/50 train/test split** (180/180) — much longer test set; one bad regime no longer dominates
6. **Bootstrap pre-training** — DNNs are warm-started on 20 synthetic DGP replications, then fine-tuned on the real series

Everything else (DGP, econometric models, ML models, plots) matches v1 so the comparison is apples-to-apples.""")

# ─── Imports ────────────────────────────────────────────────────────────────
co("""# ── Package Installation (Colab only) ──────────────────────────────────────
# !pip install -q tensorflow scikit-learn matplotlib pandas numpy seaborn statsmodels scipy""")

md("## ── SECTION 1: IMPORTS & SETUP")
co("""import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, LSTM, SimpleRNN, Dropout,
    Input, Concatenate, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
BLOCK_COLORS = {1: '#4878CF', 2: '#6ACC65', 3: '#D65F5F', 4: '#B47CC7'}

print('Libraries loaded.')
print(f'TensorFlow version: {tf.__version__}')""")

# ─── DGP ────────────────────────────────────────────────────────────────────
md("## ── SECTION 2: COMPLEX SYNTHETIC DGP (unchanged from v1)")
co("""def generate_complex_inflation_dgp(n_obs=360, seed=42):
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start='1995-01-01', periods=n_obs, freq='MS')
    t     = np.linspace(0, 1, n_obs)

    P = np.array([[0.93, 0.06, 0.01],
                  [0.05, 0.88, 0.07],
                  [0.02, 0.08, 0.90]])
    mu_r  = np.array([ 2.5,  7.0, 16.0])
    vol_r = np.array([ 0.6,  2.0,  5.5])

    regimes    = np.zeros(n_obs, dtype=int)
    regimes[0] = 0
    for i in range(1, n_obs):
        regimes[i] = rng.choice(3, p=P[regimes[i-1]])

    h = np.zeros(n_obs)
    for i in range(1, n_obs):
        h[i] = 0.97 * h[i-1] + rng.normal(0, 0.15)
    sigma_t = vol_r[regimes] * np.exp(0.5 * h)

    gdp_growth    = 3.5 + 1.8*np.sin(8*np.pi*t) + rng.standard_t(5, n_obs)*0.5
    money_growth  = 12.0 + np.cumsum(rng.normal(0, 0.1, n_obs)) + rng.normal(0, 0.8, n_obs)
    exchange_rate = rng.standard_t(4, n_obs) * 2.0
    oil_price     = 8.0*np.sin(5*np.pi*t) + rng.standard_t(3, n_obs)*4.0
    interest_rate = mu_r[regimes]*0.4 + 2.5 + rng.normal(0, 0.6, n_obs)
    fiscal_def    = -3.0 + 1.5*np.sin(3*np.pi*t) + rng.normal(0, 0.4, n_obs)

    n_j   = rng.poisson(0.04 * n_obs)
    j_idx = rng.choice(n_obs, n_j, replace=False)
    jumps = np.zeros(n_obs)
    jumps[j_idx] = rng.choice([-1, 1], n_j) * rng.exponential(5.0, n_j)
    innov = rng.standard_t(3, n_obs) * sigma_t

    money_eff = np.where(regimes == 2, 0.45, 0.20) * money_growth
    fx_eff    = np.where(exchange_rate > 1.5,
                          0.35 * exchange_rate, -0.08 * exchange_rate)
    oil_eff   = 0.30 * oil_price / (1 + np.exp(-0.25 * oil_price))
    gdp_eff   = np.where(gdp_growth < 0,
                          -0.35 * gdp_growth, -0.15 * gdp_growth)

    inflation = (mu_r[regimes]
                 + money_eff + oil_eff + fx_eff + gdp_eff
                 - 0.10 * interest_rate + 0.07 * fiscal_def
                 + jumps + innov)
    inflation = np.clip(inflation, -8, 65)

    df = pd.DataFrame({
        'inflation':    inflation,
        'gdp_growth':   gdp_growth,
        'money_growth': money_growth,
        'exchange_rate':exchange_rate,
        'oil_price':    oil_price,
        'interest_rate':interest_rate,
        'fiscal_def':   fiscal_def,
        'regime':       regimes,
    }, index=dates)
    return df

df = generate_complex_inflation_dgp(seed=SEED)
print(f'DGP generated: {len(df)} obs, {df.index[0].date()} to {df.index[-1].date()}')""")

# ─── Feature engineering: NEW for v2 (diff money, vol covariate) ────────────
md("""## ── SECTION 3: FEATURE ENGINEERING (NEW IN V2)

Two changes:
- **`d_money_growth`** = first difference of `money_growth` (the level has a unit root from `cumsum`)
- **`vol_12m`** = 12-month rolling std of inflation, *lagged by one period* to avoid look-ahead bias

These give the DNNs (and ML models) the same volatility information that SV-aware Block-2 models construct internally.""")

co("""# First-difference money_growth (it's a random walk in the DGP — unit root)
df['d_money_growth'] = df['money_growth'].diff().fillna(0.0)

# Rolling 12-month std of inflation, lagged 1 to avoid look-ahead
df['vol_12m'] = (df['inflation']
                 .rolling(window=12, min_periods=1)
                 .std()
                 .shift(1)
                 .bfill())          # backfill the first NaN with the next value

print(df[['inflation', 'money_growth', 'd_money_growth', 'vol_12m']].head(15).round(3))
print()
print('Summary:')
print(df[['d_money_growth', 'vol_12m']].describe().round(3))""")

# ─── EDA ────────────────────────────────────────────────────────────────────
md("## ── SECTION 4: STYLIZED FACTS (EDA)")
co("""fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Stylized Facts of the Synthetic Inflation DGP', fontsize=14, fontweight='bold')

REGIME_COLORS = ['#4878CF', '#6ACC65', '#D65F5F']
REGIME_LABELS = ['Low', 'Medium', 'High']

ax = axes[0, 0]
for r in range(3):
    mask = df['regime'] == r
    ax.scatter(df.index[mask], df['inflation'][mask],
               c=REGIME_COLORS[r], s=4, label=f'Regime {r} ({REGIME_LABELS[r]})', alpha=0.8)
ax.plot(df.index, df['inflation'], color='black', lw=0.4, alpha=0.4)
ax.set_title('Inflation Time Series by Regime'); ax.set_ylabel('Inflation (%)')
ax.legend(markerscale=3, fontsize=8); ax.set_xlabel('')

ax = axes[0, 1]
inf_vals = df['inflation'].values
ax.hist(inf_vals, bins=40, density=True, color='steelblue', alpha=0.5, label='Data')
kde_x = np.linspace(inf_vals.min(), inf_vals.max(), 300)
kde = stats.gaussian_kde(inf_vals)
ax.plot(kde_x, kde(kde_x), 'b-', lw=2, label='KDE')
mu_n, sig_n = inf_vals.mean(), inf_vals.std()
ax.plot(kde_x, stats.norm.pdf(kde_x, mu_n, sig_n), 'r--', lw=2, label='Normal fit')
ax.set_title('Distribution: Heavy Tails vs Normal')
ax.set_xlabel('Inflation (%)'); ax.set_ylabel('Density'); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(df.index, df['vol_12m'], color='darkorange', lw=1.5)
ax.fill_between(df.index, 0, df['vol_12m'], alpha=0.3, color='darkorange')
ax.set_title('12-Month Rolling Std (now an input to DNNs)')
ax.set_ylabel('Std Dev')

ax = axes[1, 1]
ax.plot(df.index, df['money_growth'], color='steelblue', lw=1.2, label='money_growth (level — unit root)')
ax.set_ylabel('Level', color='steelblue')
ax2 = ax.twinx()
ax2.plot(df.index, df['d_money_growth'], color='darkred', lw=1.0, alpha=0.7,
         label='d_money_growth (stationary)')
ax2.set_ylabel('First difference', color='darkred')
ax.set_title('Money Growth: Level vs First Difference')

ax = axes[2, 0]
plot_acf(df['inflation'], lags=24, ax=ax, color='steelblue',
         vlines_kwargs={'colors': 'steelblue'})
ax.set_title('ACF of Inflation (first 24 lags)'); ax.set_xlabel('Lag (months)')

ax = axes[2, 1]
corr_cols = ['inflation', 'gdp_growth', 'd_money_growth', 'exchange_rate',
             'oil_price', 'interest_rate', 'fiscal_def', 'vol_12m']
corr_mat = df[corr_cols].corr()
sns.heatmap(corr_mat, ax=ax, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, annot_kws={'size': 7}, linewidths=0.5)
ax.set_title('Correlation Heatmap (v2 features)')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig('inflation_v2_eda.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_eda.png')""")

# ─── Shared preprocessing ───────────────────────────────────────────────────
md("""## ── SECTION 5: SHARED PREPROCESSING

**Key v2 changes:**
- `FEATURE_COLS` swaps `money_growth` → `d_money_growth` and adds `vol_12m` (7 features now)
- `SPLIT = int(0.50 * len(df)) = 180`  →  much longer test set (180 obs instead of 72)
- `VAR_COLS` also uses `d_money_growth` for the VAR/TVP-VAR family""")

co("""FEATURE_COLS = ['gdp_growth', 'd_money_growth', 'exchange_rate',
                'oil_price', 'interest_rate', 'fiscal_def', 'vol_12m']
TARGET_COL   = 'inflation'

SPLIT        = int(0.50 * len(df))    # 180 train / 180 test  (v2: was 0.80)
SEQ_LEN      = 12

y_full     = df[TARGET_COL].values
X_full     = df[FEATURE_COLS].values
n_test     = len(y_full) - SPLIT
test_idx   = df.index[SPLIT:]

y_train_raw = y_full[:SPLIT]
y_test_raw  = y_full[SPLIT:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X_full[:SPLIT])
scaler_y.fit(y_full[:SPLIT].reshape(-1, 1))

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(y)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def eval_metrics(actual, predicted, name):
    return {'Model': name,
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'MAE':  float(mean_absolute_error(actual, predicted)),
            'R2':   float(r2_score(actual, predicted))}

def diebold_mariano(e1, e2, h=1):
    d = e1**2 - e2**2
    n, d_bar = len(d), d.mean()
    dc  = d - d_bar
    lrv = float(np.dot(dc, dc)) / n
    for k in range(1, h):
        lrv += 2*(1 - k/h) * float(np.dot(dc[k:], dc[:-k])) / n
    if lrv <= 0:
        return np.nan, np.nan
    dm = (d_bar / np.sqrt(lrv / n)) * np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n-1))

all_preds = {}

print(f'Features    : {FEATURE_COLS}')
print(f'Train       : {SPLIT} obs  ({df.index[0].date()} – {df.index[SPLIT-1].date()})')
print(f'Test        : {n_test} obs ({test_idx[0].date()} – {test_idx[-1].date()})')
print(f'Test/Train  : 50/50')""")

# ─── BLOCK 1 ────────────────────────────────────────────────────────────────
md("## ── SECTION 6: BLOCK 1 — TRADITIONAL ECONOMETRIC MODELS")
co("""# ── 6.1 Random Walk ───────────────────────────────────────────────────────
rw_preds = y_full[SPLIT-1:SPLIT+n_test-1].copy()
all_preds['RW'] = rw_preds
print(f"{'RW':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, rw_preds)):.4f}")""")

co("""# ── 6.2 AR(p) — AIC lag selection, expanding window ──────────────────────
print('Fitting AR(p) with AIC lag selection...')
aic_ar = {lag: AutoReg(y_full[:SPLIT], lags=lag, old_names=False).fit().aic
          for lag in range(1, 13)}
ar_lag = min(aic_ar, key=aic_ar.get)
print(f'  Selected lag order: {ar_lag}')

ar_preds = []
for t in range(n_test):
    m = AutoReg(y_full[:SPLIT+t], lags=ar_lag, old_names=False).fit()
    ar_preds.append(float(m.forecast(steps=1)))
all_preds['AR'] = np.array(ar_preds)
print(f"{'AR':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['AR'])):.4f}")""")

co("""# ── 6.3 ARIMA(2,0,1) — expanding window ──────────────────────────────────
def _fc_to_float(fc):
    \"\"\"Robust to statsmodels returning Series or ndarray.\"\"\"
    if hasattr(fc, 'iloc'):
        return float(fc.iloc[0])
    arr = np.asarray(fc).ravel()
    return float(arr[0])

print('Fitting ARIMA(2,0,1) expanding window...')
arima_preds = []
for t in range(n_test):
    try:
        m = ARIMA(y_full[:SPLIT+t], order=(2, 0, 1)).fit()
        arima_preds.append(_fc_to_float(m.forecast(steps=1)))
    except Exception:
        arima_preds.append(arima_preds[-1] if arima_preds else float(y_full[SPLIT+t-1]))
all_preds['ARIMA'] = np.array(arima_preds)
print(f"{'ARIMA':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['ARIMA'])):.4f}")""")

co("""# ── 6.4 ARIMAX — ARIMA(2,0,1) with exogenous regressors ──────────────────
print('Fitting ARIMAX(2,0,1) expanding window...')
arimax_preds = []
for t in range(n_test):
    try:
        m = ARIMA(y_full[:SPLIT+t], order=(2, 0, 1),
                  exog=X_full[:SPLIT+t]).fit()
        fc = m.forecast(steps=1, exog=X_full[SPLIT+t:SPLIT+t+1])
        arimax_preds.append(_fc_to_float(fc))
    except Exception:
        arimax_preds.append(arimax_preds[-1] if arimax_preds else float(y_full[SPLIT+t-1]))
all_preds['ARIMAX'] = np.array(arimax_preds)
print(f"{'ARIMAX':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['ARIMAX'])):.4f}")

print()
print('── Block 1 Summary ──────────────────────────────────────')
for m in ['RW', 'AR', 'ARIMA', 'ARIMAX']:
    r = np.sqrt(mean_squared_error(y_test_raw, all_preds[m]))
    print(f'  {m:10s}  RMSE = {r:.4f}')""")

# ─── BLOCK 2 ────────────────────────────────────────────────────────────────
md("## ── SECTION 7: BLOCK 2 — NONLINEAR ECONOMETRIC MODELS")
co("""# ── Kalman Filter core for TVP-AR and AR-SV ───────────────────────────────
def _tvp_ar_core(y, train_end, p=2, Q_scale=1e-4, sv_lambda=None):
    n_train = train_end
    n_total = len(y)
    n_test_  = n_total - n_train

    X_ols = np.column_stack([np.ones(n_train - p)] +
                             [y[i:n_train-p+i] for i in range(p)])
    y_ols = y[p:n_train]
    beta0, *_ = np.linalg.lstsq(X_ols, y_ols, rcond=None)

    beta_hat = beta0.copy()
    P_cov    = np.eye(p + 1) * 1.0
    Q        = np.eye(p + 1) * Q_scale
    R        = float(np.var(y_ols - X_ols @ beta0))

    for i in range(p, n_train):
        x_t = np.array([1.0] + list(y[i-p:i][::-1]))
        P_pred = P_cov + Q
        S   = float(x_t @ P_pred @ x_t) + R
        K   = P_pred @ x_t / S
        err = y[i] - float(x_t @ beta_hat)
        beta_hat = beta_hat + K * err
        P_cov    = (np.eye(p + 1) - np.outer(K, x_t)) @ P_pred
        if sv_lambda is not None:
            R = sv_lambda * R + (1 - sv_lambda) * err**2

    forecasts   = np.zeros(n_test_)
    sigma2_path = np.zeros(n_test_)
    for t in range(n_test_):
        idx = n_train + t
        x_t = np.array([1.0] + list(y[idx-p:idx][::-1]))
        sigma2_path[t] = R
        forecasts[t]   = float(x_t @ beta_hat)
        P_pred   = P_cov + Q
        S        = float(x_t @ P_pred @ x_t) + R
        K        = P_pred @ x_t / S
        err      = y[idx] - forecasts[t]
        beta_hat = beta_hat + K * err
        P_cov    = (np.eye(p + 1) - np.outer(K, x_t)) @ P_pred
        if sv_lambda is not None:
            R = sv_lambda * R + (1 - sv_lambda) * err**2
    return forecasts, sigma2_path""")

co("""# ── 7.1 AR-SV ────────────────────────────────────────────────────────────
print('Fitting AR-SV (Kalman + EWMA SV)...')
arsv_preds, arsv_sig2 = _tvp_ar_core(y_full, SPLIT, p=2, sv_lambda=0.94)
all_preds['AR-SV'] = arsv_preds
print(f"{'AR-SV':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['AR-SV'])):.4f}")

# ── 7.2 TVP-AR ───────────────────────────────────────────────────────────
print('Fitting TVP-AR (Kalman, no SV)...')
tvpar_preds, _ = _tvp_ar_core(y_full, SPLIT, p=2, sv_lambda=None)
all_preds['TVP-AR'] = tvpar_preds
print(f"{'TVP-AR':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['TVP-AR'])):.4f}")""")

co("""# ── 7.3 MS-AR (Markov-Switching AR, p=2, 2 regimes) ──────────────────────
# v2 change vs v1: dropped from 3 → 2 regimes because the v2 train sample is
# halved (180 obs vs 288), and a 3-regime × switching-AR model is no longer
# identifiable — it converges to non-stationary parameter blobs.
# Strategy: fit once on training data with multi-start; for each test step,
# re-FILTER (not re-fit) on the extended history to update regime probabilities,
# then compute the one-step-ahead forecast as a prob-weighted average of
# regime-specific AR(2) predictions. Exact under the fitted-parameter
# assumption and avoids the `predict(start>nobs)` NotImplementedError in
# current statsmodels.
print('Fitting MS-AR(2 regimes, p=2)...')
MSAR_K, MSAR_P = 2, 2

def _msar_extract(params, k_regimes=MSAR_K, order=MSAR_P):
    \"\"\"Pull regime-specific (const, AR coefs) from a fitted MS-AR params Series.\"\"\"
    const = np.array([float(params[f'const[{k}]']) for k in range(k_regimes)])
    ar = np.zeros((k_regimes, order))
    for k in range(k_regimes):
        for l in range(1, order + 1):
            ar[k, l - 1] = float(params[f'ar.L{l}[{k}]'])
    return const, ar

def _msar_forecast_one(y_hist, ms_fit, trans_mat, const, ar_coefs, switching_ar):
    \"\"\"One-step-ahead forecast given full history; trans_mat[i,j]=P(s_{t+1}=i|s_t=j).\"\"\"
    m_ext = MarkovAutoregression(y_hist, k_regimes=MSAR_K, order=MSAR_P,
                                  switching_ar=switching_ar)
    res   = m_ext.filter(ms_fit.params)
    p_T   = np.asarray(res.filtered_marginal_probabilities)[-1]   # (k,)
    p_next = trans_mat @ p_T                                       # (k,)
    last_y, lag_y = y_hist[-1], y_hist[-2]
    fc_by_reg = const + ar_coefs[:, 0] * last_y + ar_coefs[:, 1] * lag_y
    return float(p_next @ fc_by_reg)

def _fit_msar_robust(y_train, k=3, p=2, n_inits=12, seed=0):
    \"\"\"Try multiple random inits; keep the fit with the highest log-likelihood
    whose AR coefficients are stationary in every regime (sum |L1|+|L2| < 1).\"\"\"
    best = None
    rng = np.random.default_rng(seed)
    for i in range(n_inits):
        try:
            mod = MarkovAutoregression(y_train, k_regimes=k, order=p,
                                       switching_ar=True)
            if i == 0:
                res = mod.fit(disp=False, maxiter=200)
            else:
                # Random starts around defaults
                start = mod.start_params + rng.normal(0, 0.1, size=mod.start_params.shape)
                res = mod.fit(start_params=start, disp=False, maxiter=200)
            # Validate stationarity per regime
            pdict_ = dict(zip(res.model.param_names, np.asarray(res.params)))
            ar_sum_ok = True
            for kk in range(k):
                s = sum(abs(pdict_[f'ar.L{l}[{kk}]']) for l in range(1, p+1))
                if s >= 0.99:
                    ar_sum_ok = False; break
            if not ar_sum_ok:
                continue
            if best is None or res.llf > best.llf:
                best = res
        except Exception:
            continue
    if best is None:
        # Last-resort: fit without switching the AR coefficients (fewer params)
        try:
            best = MarkovAutoregression(y_train, k_regimes=k, order=p,
                                         switching_ar=False).fit(disp=False, maxiter=200)
        except Exception:
            best = None
    return best

print('Fitting MS-AR(3 regimes, p=2) — robust multi-start...')
ms_fit = _fit_msar_robust(y_full[:SPLIT], k=MSAR_K, p=MSAR_P)
if ms_fit is None:
    print('  MS-AR all inits failed; falling back to AR(2) constant forecast.')
    ar2 = AutoReg(y_full[:SPLIT], lags=2, old_names=False).fit()
    all_preds['MS-AR'] = np.array([float(ar2.forecast(steps=1)[0])] * n_test)
else:
    rt = np.asarray(ms_fit.regime_transition)
    trans_mat = rt[..., 0] if rt.ndim == 3 else rt
    pdict = dict(zip(ms_fit.model.param_names, np.asarray(ms_fit.params)))
    has_switch_ar = any('ar.L1[1]' in nm for nm in ms_fit.model.param_names)
    const    = np.array([pdict[f'const[{k}]'] for k in range(MSAR_K)])
    if has_switch_ar:
        ar_coefs = np.array([[pdict[f'ar.L{l}[{k}]'] for l in range(1, MSAR_P+1)]
                              for k in range(MSAR_K)])
    else:
        # Non-switching AR: same coefs for every regime
        ar_shared = np.array([pdict[f'ar.L{l}'] for l in range(1, MSAR_P+1)])
        ar_coefs  = np.tile(ar_shared, (MSAR_K, 1))
    print(f'  MS-AR fitted (switching_ar={has_switch_ar}, llf={ms_fit.llf:.2f})')
    print(f'  const per regime: {const.round(3)}')
    print(f'  AR sum |L1|+|L2| per regime: {[round(float(np.sum(np.abs(r))),3) for r in ar_coefs]}')

    ms_preds = []
    for t in range(n_test):
        try:
            ms_preds.append(_msar_forecast_one(y_full[:SPLIT + t], ms_fit,
                                                trans_mat, const, ar_coefs,
                                                switching_ar=has_switch_ar))
        except Exception:
            ms_preds.append(ms_preds[-1] if ms_preds else float(y_full[SPLIT + t - 1]))
    all_preds['MS-AR'] = np.array(ms_preds)
print(f"{'MS-AR':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['MS-AR'])):.4f}")""")

co("""# ── 7.4 FA-AR (Factor-Augmented AR) ──────────────────────────────────────
# BUG-FIX vs v1: previous loop `for i in range(p_fa,0,-1): y_av[i:n_av-p_fa+i]`
# put y_av[p_fa:n_av] (= y_t itself) into the regressors when i=p_fa, so OLS
# returned alpha=1 on that column and FA-AR collapsed to a pure random-walk.
# Correct lag construction:
#   lag-l column for predicting y[k+p] is y[k+p-l] for k=0..n_av-p-1
#   => y_av[p_fa-l : n_av-l]   for l=1..p_fa
print('Fitting FA-AR (PCA factors + AR lags, expanding window)...')
pca = PCA(n_components=2)
pca.fit(X_full[:SPLIT])
factors_all = pca.transform(X_full)

fa_ar_preds = []
p_fa = 2
for t in range(n_test):
    y_av = y_full[:SPLIT+t]
    f_av = factors_all[:SPLIT+t]
    try:
        n_av = len(y_av)
        # lag-l regressor:  y_av[p_fa-l : n_av-l]  for l=1..p_fa
        X_fit = np.column_stack(
            [y_av[p_fa-l : n_av-l] for l in range(1, p_fa+1)] + [f_av[p_fa:n_av]]
        )
        y_fit = y_av[p_fa:]
        coef = np.linalg.lstsq(
            np.column_stack([np.ones(len(y_fit)), X_fit]), y_fit, rcond=None
        )[0]
        # x_next mirrors design: [1, y_{t-1}, y_{t-2}, ..., f_t]
        x_next = np.array([1.0] +
                          list(y_av[-p_fa:][::-1]) +
                          list(factors_all[SPLIT+t]))
        fa_ar_preds.append(float(x_next @ coef))
    except Exception:
        fa_ar_preds.append(float(y_av[-1]))

all_preds['FA-AR'] = np.array(fa_ar_preds)
print(f"{'FA-AR':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['FA-AR'])):.4f}")""")

co("""# ── 7.5 TVP-VAR (rolling 60-month window) ────────────────────────────────
print('Fitting TVP-VAR (rolling 60-month window)...')
VAR_COLS = ['inflation', 'gdp_growth', 'd_money_growth', 'interest_rate']
data_var = df[VAR_COLS].values
ROLL_WIN, TVP_P = 60, 2

tvpvar_preds = []
for t in range(SPLIT, SPLIT+n_test):
    sub = data_var[max(0, t-ROLL_WIN):t]
    try:
        m  = VAR(sub).fit(maxlags=TVP_P, ic=None)
        fc = m.forecast(sub[-m.k_ar:], steps=1)
        tvpvar_preds.append(float(fc[0, 0]))
    except Exception:
        tvpvar_preds.append(float(y_full[t-1]))
all_preds['TVP-VAR'] = np.array(tvpvar_preds)
print(f"{'TVP-VAR':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['TVP-VAR'])):.4f}")

# ── 7.6 TVP-VAR-SV ───────────────────────────────────────────────────────
print('Fitting TVP-VAR-SV (rolling window + EWMA SV)...')
LAM = 0.94
_m0  = VAR(data_var[:SPLIT]).fit(maxlags=TVP_P, ic=None)
_Rsv = float(np.var(_m0.resid[:, 0]))
tvpvar_sv_preds = []
for t in range(SPLIT, SPLIT+n_test):
    sub = data_var[max(0, t-ROLL_WIN):t]
    try:
        m  = VAR(sub).fit(maxlags=TVP_P, ic=None)
        fc = m.forecast(sub[-m.k_ar:], steps=1)
        pt = float(fc[0, 0])
    except Exception:
        pt = float(y_full[t-1])
    tvpvar_sv_preds.append(pt)
    _Rsv = LAM * _Rsv + (1 - LAM) * (y_full[t] - pt)**2
all_preds['TVP-VAR-SV'] = np.array(tvpvar_sv_preds)
print(f"{'TVP-VAR-SV':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['TVP-VAR-SV'])):.4f}")

# ── 7.7 VAR-SV (fixed VAR + EWMA SV) ─────────────────────────────────────
print('Fitting VAR-SV (fixed VAR + EWMA SV)...')
var_fixed = VAR(data_var[:SPLIT]).fit(maxlags=2, ic='aic')
_Rsv2 = float(np.var(var_fixed.resid[:, 0]))
varsv_preds = []
for t in range(SPLIT, SPLIT+n_test):
    window = data_var[max(0, t-var_fixed.k_ar):t]
    try:
        fc = var_fixed.forecast(window[-var_fixed.k_ar:], steps=1)
        pt = float(fc[0, 0])
    except Exception:
        pt = float(y_full[t-1])
    varsv_preds.append(pt)
    _Rsv2 = LAM * _Rsv2 + (1 - LAM) * (y_full[t] - pt)**2
all_preds['VAR-SV'] = np.array(varsv_preds)
print(f"{'VAR-SV':12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds['VAR-SV'])):.4f}")

print()
print('── Block 2 Summary ──────────────────────────────────────')
for m in ['AR-SV', 'MS-AR', 'TVP-AR', 'FA-AR', 'TVP-VAR', 'TVP-VAR-SV', 'VAR-SV']:
    r = np.sqrt(mean_squared_error(y_test_raw, all_preds[m]))
    print(f'  {m:12s}  RMSE = {r:.4f}')""")

# ─── BLOCK 3 ────────────────────────────────────────────────────────────────
md("## ── SECTION 8: BLOCK 3 — MACHINE LEARNING MODELS")
co("""def make_ml_features(y, X, p=12):
    rows = []
    for t in range(p, len(y)):
        lag_y = y[t-p:t][::-1]
        rows.append(np.concatenate([lag_y, X[t]]))
    return np.array(rows), y[p:]

ML_P = 12
Xml_full, yml_full = make_ml_features(y_full, X_full, p=ML_P)
sc_ml = StandardScaler().fit(Xml_full[:SPLIT - ML_P])

print(f'ML feature matrix shape: {Xml_full.shape}  ({ML_P} AR lags + {X_full.shape[1]} macro features)')

for ModelClass, name, kwargs in [
    (LassoCV,      'Lasso',      {'cv': 5, 'max_iter': 5000}),
    (RidgeCV,      'Ridge',      {'alphas': np.logspace(-3, 3, 20)}),
    (ElasticNetCV, 'ElasticNet', {'cv': 5, 'max_iter': 5000,
                                  'l1_ratio': [.1, .5, .7, .9, .95, 1]}),
]:
    preds = []
    for t in range(n_test):
        avail_end = SPLIT + t
        Xav  = Xml_full[:avail_end - ML_P]
        yav  = yml_full[:avail_end - ML_P]
        if len(Xav) < 20:
            preds.append(float(y_full[avail_end - 1])); continue
        Xav_sc = sc_ml.transform(Xav)
        m_ml   = ModelClass(**kwargs).fit(Xav_sc, yav)
        x_next = sc_ml.transform(Xml_full[avail_end - ML_P:avail_end - ML_P + 1])
        preds.append(float(m_ml.predict(x_next)[0]))
    all_preds[name] = np.array(preds)
    print(f"{name:12s}  RMSE = {np.sqrt(mean_squared_error(y_test_raw, all_preds[name])):.4f}")

print()
print('── Block 3 Summary ──────────────────────────────────────')
for m in ['Lasso', 'Ridge', 'ElasticNet']:
    r = np.sqrt(mean_squared_error(y_test_raw, all_preds[m]))
    print(f'  {m:12s}  RMSE = {r:.4f}')""")

# ─── BLOCK 4 with the 4 DNN-specific upgrades ───────────────────────────────
md("""## ── SECTION 9: BLOCK 4 — DEEP NEURAL NETWORKS (V2 upgrades concentrated here)

What changes versus v1:
1. **5 random seeds per architecture**, predictions averaged → reduces single-seed noise
2. **Huber loss (δ=1.0)** → robust to Student-t(3) shocks and Poisson jumps
3. **Bootstrap pre-training** on 20 synthetic DGP replications → warm-starts each architecture before fine-tuning
4. Inputs include the new `d_money_growth` and `vol_12m` features (7 features total)""")

co("""# ── 9.1 DNN preprocessing (sequences for the REAL data) ──────────────────
X_sc_all = scaler_X.transform(X_full)
y_sc_all = scaler_y.transform(y_full.reshape(-1, 1)).ravel()

X_seq, y_seq = make_sequences(X_sc_all, y_sc_all, SEQ_LEN)
SPLIT_SEQ    = SPLIT - SEQ_LEN
X_seq_train  = X_seq[:SPLIT_SEQ]
y_seq_train  = y_seq[:SPLIT_SEQ]
X_seq_test   = X_seq[SPLIT_SEQ:]
y_seq_test   = y_seq[SPLIT_SEQ:]
N_FEAT       = X_seq.shape[2]

# Current-period features for FF-LSTM hybrid
X_cur_train  = X_sc_all[SEQ_LEN:SPLIT]
X_cur_test   = X_sc_all[SPLIT:]

def inverse_scale_y(y_sc):
    return scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()

print(f'Real sequences      : {X_seq.shape}')
print(f'Train / Test seqs   : {X_seq_train.shape[0]} / {X_seq_test.shape[0]}')
print(f'N features per step : {N_FEAT}')""")

co("""# ── 9.2 Bootstrap pre-training pool: 20 synthetic DGP replications ───────
N_BOOTSTRAP    = 20
BOOTSTRAP_BASE = 1000

def make_bootstrap_pool(n_reps=N_BOOTSTRAP, base_seed=BOOTSTRAP_BASE,
                       split=SPLIT, seq_len=SEQ_LEN):
    Xs, ys = [], []
    Xcur_, ycur_ = [], []
    for r in range(n_reps):
        df_r = generate_complex_inflation_dgp(seed=base_seed + r)
        df_r['d_money_growth'] = df_r['money_growth'].diff().fillna(0.0)
        df_r['vol_12m']        = (df_r['inflation']
                                   .rolling(12, min_periods=1).std()
                                   .shift(1).bfill())
        X_r = df_r[FEATURE_COLS].values[:split]
        y_r = df_r[TARGET_COL].values[:split]
        # Scale with the MAIN scalers (real-data fit) so models see consistent inputs
        X_r_sc = scaler_X.transform(X_r)
        y_r_sc = scaler_y.transform(y_r.reshape(-1, 1)).ravel()
        Xs_r, ys_r = make_sequences(X_r_sc, y_r_sc, seq_len)
        Xs.append(Xs_r); ys.append(ys_r)
        Xcur_.append(X_r_sc[seq_len:]); ycur_.append(ys_r)
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(Xcur_), np.concatenate(ycur_))

X_pre, y_pre, X_pre_cur, y_pre_cur = make_bootstrap_pool()
print(f'Bootstrap pool       : {N_BOOTSTRAP} reps  →  {X_pre.shape[0]} sequences')
print(f'Real fine-tune set   : {X_seq_train.shape[0]} sequences')""")

co("""# ── 9.3 DNN builder functions (Huber loss, identical architectures to v1) ─
HUBER = Huber(delta=1.0)

def build_lstm(seq_len, n_feat, lr=3e-4):
    m = Sequential([
        LSTM(64, input_shape=(seq_len, n_feat), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ], name='LSTM')
    m.compile(optimizer=Adam(lr), loss=HUBER); return m

def build_rnn(seq_len, n_feat, lr=3e-4):
    m = Sequential([
        SimpleRNN(64, input_shape=(seq_len, n_feat), return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1),
    ], name='RNN')
    m.compile(optimizer=Adam(lr), loss=HUBER); return m

class JordanCell(tf.keras.layers.Layer):
    def __init__(self, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.W_h   = Dense(hidden_units, activation='tanh',   name='hidden')
        self.W_out = Dense(1,            activation='linear', name='output')
    @property
    def state_size(self): return 1
    @property
    def output_size(self): return 1
    def call(self, inputs, states):
        context = states[0]
        x = tf.concat([inputs, context], axis=-1)
        h = self.W_h(x); out = self.W_out(h)
        return out, [out]
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros((batch_size, 1), dtype=dtype or tf.float32)]

def build_jnn(seq_len, n_feat, lr=3e-4):
    inp  = Input(shape=(seq_len, n_feat))
    cell = JordanCell(64)
    rnn  = tf.keras.layers.RNN(cell, return_sequences=False, name='jordan_rnn')(inp)
    out  = Dense(16, activation='relu')(rnn)
    out  = Dense(1)(out)
    m = Model(inp, out, name='JNN'); m.compile(optimizer=Adam(lr), loss=HUBER); return m

def build_mrn(seq_len, n_feat, units=32, lr=3e-4):
    inp = Input(shape=(seq_len, n_feat))
    b_short = tf.keras.layers.Lambda(lambda x: x[:, -3:, :])(inp)
    b_short = LSTM(units)(b_short)
    b_med   = LSTM(units)(inp)
    b_long  = tf.keras.layers.Lambda(lambda x: x[:, ::2, :])(inp)
    b_long  = LSTM(units)(b_long)
    merged  = Concatenate()([b_short, b_med, b_long])
    x       = Dense(32, activation='relu')(merged)
    x       = Dropout(0.2)(x)
    out     = Dense(1)(x)
    m = Model(inp, out, name='MRN'); m.compile(optimizer=Adam(lr), loss=HUBER); return m

def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2*(dims//2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

def build_transformer(seq_len, n_feat, d_model=32, n_heads=2, ff_dim=64, lr=1e-3):
    inp = Input(shape=(seq_len, n_feat))
    x   = Dense(d_model)(inp)
    x   = x + positional_encoding(seq_len, d_model)
    a   = MultiHeadAttention(num_heads=n_heads,
                             key_dim=d_model // n_heads, dropout=0.1)(x, x)
    x   = LayerNormalization(epsilon=1e-6)(x + a)
    f   = Dense(ff_dim, activation='relu')(x); f = Dense(d_model)(f)
    x   = LayerNormalization(epsilon=1e-6)(x + f)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(32, activation='relu')(x); x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out, name='Transformer'); m.compile(optimizer=Adam(lr), loss=HUBER); return m

def build_ff_lstm(seq_len, n_feat, lr=3e-4):
    seq_inp = Input(shape=(seq_len, n_feat), name='seq_input')
    cur_inp = Input(shape=(n_feat,),         name='cur_input')
    h       = LSTM(64)(seq_inp); h = Dropout(0.2)(h)
    g       = Dense(32, activation='relu')(cur_inp); g = Dropout(0.2)(g)
    merged  = Concatenate()([h, g])
    x       = Dense(32, activation='relu')(merged)
    out     = Dense(1)(x)
    m = Model([seq_inp, cur_inp], out, name='FF-LSTM'); m.compile(optimizer=Adam(lr), loss=HUBER); return m

print('DNN builders defined (Huber loss).')""")

co("""# ── 9.4 Multi-seed training loop with bootstrap pre-training ─────────────
SEEDS_DNN     = [42, 43, 44, 45, 46]    # 5 seeds
PRETRAIN_EPS  = 25                       # epochs on bootstrap pool
FT_EPS        = 200                      # max fine-tune epochs (early stopping)
BATCH         = 64
FT_LR         = 1e-4                     # lower LR for fine-tuning

def train_one_dnn(name, build_fn, seed, two_input=False):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed); np.random.seed(seed)
    model = build_fn(SEQ_LEN, N_FEAT)

    # ── (a) Pre-train on bootstrap pool ─────────────────────────────────
    if two_input:
        model.fit([X_pre, X_pre_cur], y_pre,
                  epochs=PRETRAIN_EPS, batch_size=BATCH,
                  validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=6,
                                            restore_best_weights=True)])
    else:
        model.fit(X_pre, y_pre,
                  epochs=PRETRAIN_EPS, batch_size=BATCH,
                  validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=6,
                                            restore_best_weights=True)])

    # ── (b) Drop LR and fine-tune on real data ──────────────────────────
    # Keras 3: easiest portable way to switch LR is recompile with fresh optimizer
    model.compile(optimizer=Adam(FT_LR), loss=HUBER)
    cbks = [EarlyStopping(monitor='val_loss', patience=25,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-6)]
    if two_input:
        model.fit([X_seq_train, X_cur_train], y_seq_train,
                  epochs=FT_EPS, batch_size=BATCH,
                  validation_split=0.15, verbose=0, callbacks=cbks)
        pred_sc = model.predict([X_seq_test, X_cur_test], verbose=0).ravel()
    else:
        model.fit(X_seq_train, y_seq_train,
                  epochs=FT_EPS, batch_size=BATCH,
                  validation_split=0.15, verbose=0, callbacks=cbks)
        pred_sc = model.predict(X_seq_test, verbose=0).ravel()

    return inverse_scale_y(pred_sc)

def train_dnn_multiseed(name, build_fn, two_input=False):
    print(f'  {name}: training across {len(SEEDS_DNN)} seeds ...')
    seed_preds = []
    seed_rmses = []
    for s in SEEDS_DNN:
        yp = train_one_dnn(name, build_fn, s, two_input=two_input)
        seed_preds.append(yp)
        r = float(np.sqrt(mean_squared_error(y_test_raw, yp)))
        seed_rmses.append(r)
        print(f'     seed={s}  RMSE={r:.4f}')
    mean_pred = np.mean(seed_preds, axis=0)
    rmse_mean = float(np.sqrt(mean_squared_error(y_test_raw, mean_pred)))
    print(f'     ENSEMBLE RMSE = {rmse_mean:.4f}  (seed mean={np.mean(seed_rmses):.4f}, '
          f'std={np.std(seed_rmses):.4f})')
    return mean_pred, seed_rmses

dnn_seed_diagnostics = {}

print('Training DNN models (bootstrap pretrain → fine-tune × 5 seeds)...\\n')
for name, fn, two_in in [
    ('LSTM',        build_lstm,        False),
    ('RNN',         build_rnn,         False),
    ('JNN',         build_jnn,         False),
    ('MRN',         build_mrn,         False),
    ('Transformer', build_transformer, False),
    ('FF-LSTM',     build_ff_lstm,     True),
]:
    yhat, seed_rmses = train_dnn_multiseed(name, fn, two_input=two_in)
    all_preds[name] = yhat
    dnn_seed_diagnostics[name] = seed_rmses
    print()

print('── Block 4 Summary (ensemble RMSE) ──────────────────────')
for m in ['LSTM', 'RNN', 'JNN', 'MRN', 'Transformer', 'FF-LSTM']:
    r = np.sqrt(mean_squared_error(y_test_raw, all_preds[m]))
    print(f'  {m:14s}  RMSE = {r:.4f}')""")

co("""# ── 9.5 DNN seed-variance diagnostic ──────────────────────────────────────
seed_diag_df = pd.DataFrame({
    m: dnn_seed_diagnostics[m] for m in dnn_seed_diagnostics
}, index=[f'seed={s}' for s in SEEDS_DNN])
seed_diag_df.loc['mean']     = seed_diag_df.mean()
seed_diag_df.loc['std']      = seed_diag_df.iloc[:-1].std()
seed_diag_df.loc['ensemble'] = [
    np.sqrt(mean_squared_error(y_test_raw, all_preds[m])) for m in seed_diag_df.columns
]
print('DNN RMSE across seeds (lower is better):')
print(seed_diag_df.round(4).to_string())
print()
print('Note: ensemble RMSE is typically lower than the per-seed mean, '
      'confirming that variance reduction from averaging is genuine.')""")

# ─── Results, DM tests ──────────────────────────────────────────────────────
md("## ── SECTION 10: HORSE RACE RESULTS + DM TESTS")
co("""BLOCK_MAP = {
    'RW': 1, 'AR': 1, 'ARIMA': 1, 'ARIMAX': 1,
    'AR-SV': 2, 'MS-AR': 2, 'TVP-AR': 2, 'FA-AR': 2,
    'TVP-VAR': 2, 'TVP-VAR-SV': 2, 'VAR-SV': 2,
    'Lasso': 3, 'Ridge': 3, 'ElasticNet': 3,
    'LSTM': 4, 'RNN': 4, 'JNN': 4, 'MRN': 4, 'Transformer': 4, 'FF-LSTM': 4,
}
BLOCK_NAMES = {1: 'Block 1: Traditional', 2: 'Block 2: Nonlinear Econo.',
               3: 'Block 3: ML', 4: 'Block 4: DNN'}

rows = []
for name, preds in all_preds.items():
    m = eval_metrics(y_test_raw, preds, name)
    m['Block']      = BLOCK_MAP.get(name, 0)
    m['Block Name'] = BLOCK_NAMES.get(BLOCK_MAP.get(name, 0), 'Unknown')
    rows.append(m)

results_df = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
results_df['Rank'] = results_df.index + 1

print('=' * 75)
print('FULL HORSE RACE (v2) — SORTED BY RMSE')
print('=' * 75)
print(results_df[['Rank', 'Model', 'Block Name', 'RMSE', 'MAE', 'R2']]
      .to_string(index=False, float_format='%.4f'))

best_per_block = {}
print()
print('── Best Model Per Block ─────────────────────────────────────────────')
for blk in [1, 2, 3, 4]:
    sub = results_df[results_df['Block'] == blk]
    best_row = sub.loc[sub['RMSE'].idxmin()]
    best_per_block[blk] = best_row['Model']
    print(f'  Block {blk} ({BLOCK_NAMES[blk]}): {best_row["Model"]:14s}  RMSE = {best_row["RMSE"]:.4f}')""")

co("""# ── DM tests: best-of-block adjacent comparisons ─────────────────────────
print('=' * 65)
print('DIEBOLD-MARIANO TESTS — ADJACENT BLOCK COMPARISONS')
print('H0: equal predictive accuracy. Positive DM => challenger better.')
print('=' * 65)
for blk_inc, blk_ch in [(1, 2), (2, 3), (3, 4)]:
    m_i = best_per_block[blk_inc]; m_c = best_per_block[blk_ch]
    e_i = y_test_raw - all_preds[m_i]
    e_c = y_test_raw - all_preds[m_c]
    dm, p = diebold_mariano(e_i, e_c)
    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else 'n.s.'))
    direction = 'IMPROVEMENT' if (not np.isnan(dm) and dm > 0) else 'No sig. improvement'
    print(f'  Block {blk_inc} ({m_i}) vs Block {blk_ch} ({m_c}): '
          f'DM = {dm:+.3f}, p = {p:.4f}  {sig}  =>  {direction}')""")

co("""# ── Full pairwise DM matrix ──────────────────────────────────────────────
model_order = ['RW', 'AR', 'ARIMA', 'ARIMAX',
               'AR-SV', 'TVP-AR', 'MS-AR', 'FA-AR', 'TVP-VAR', 'TVP-VAR-SV', 'VAR-SV',
               'Lasso', 'Ridge', 'ElasticNet',
               'LSTM', 'RNN', 'JNN', 'MRN', 'Transformer', 'FF-LSTM']
model_order = [m for m in model_order if m in all_preds]
n_m = len(model_order)
dm_pval = np.full((n_m, n_m), np.nan)
for i, m1 in enumerate(model_order):
    for j, m2 in enumerate(model_order):
        if i != j:
            e1 = y_test_raw - all_preds[m1]
            e2 = y_test_raw - all_preds[m2]
            _, p = diebold_mariano(e1, e2)
            dm_pval[i, j] = p
dm_pval_df = pd.DataFrame(dm_pval, index=model_order, columns=model_order)
print('Pairwise DM matrix:', dm_pval_df.shape)""")

# ─── Plots ──────────────────────────────────────────────────────────────────
md("## ── SECTION 11: VISUALIZATIONS")
co("""# ── Figure 1: RMSE comparison bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
plot_df = results_df.sort_values('RMSE', ascending=False)
colors  = [BLOCK_COLORS[b] for b in plot_df['Block']]
bars = ax.barh(plot_df['Model'], plot_df['RMSE'], color=colors, alpha=0.85, edgecolor='none')

best_models = set(best_per_block.values())
for bar, (_, row) in zip(bars, plot_df.iterrows()):
    if row['Model'] in best_models:
        bar.set_edgecolor('black'); bar.set_linewidth(2.5); bar.set_alpha(1.0)

current_block = None
for i, (_, row) in enumerate(plot_df.iterrows()):
    if row['Block'] != current_block:
        if current_block is not None:
            ax.axhline(i - 0.5, color='gray', lw=1, ls='--', alpha=0.5)
        current_block = row['Block']

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=BLOCK_COLORS[b], label=BLOCK_NAMES[b]) for b in [1,2,3,4]]
legend_elements.append(Patch(facecolor='white', edgecolor='black', lw=2, label='Best in block'))
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

ax.set_xlabel('RMSE (lower is better)', fontsize=11)
ax.set_title('Inflation Forecast Horse Race v2 — RMSE by Model and Block',
             fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=9)
for bar, rmse in zip(bars, plot_df['RMSE']):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{rmse:.2f}', va='center', fontsize=7.5)
plt.tight_layout()
plt.savefig('inflation_v2_rmse.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_rmse.png')""")

co("""# ── Figure 2: Block-average RMSE ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
block_rmse_data = {BLOCK_NAMES[blk]:
                   results_df[results_df['Block']==blk]['RMSE'].values
                   for blk in [1,2,3,4]}
block_labels = list(block_rmse_data.keys())
block_vals   = list(block_rmse_data.values())
block_cols   = [BLOCK_COLORS[b] for b in [1,2,3,4]]

ax = axes[0]
vp = ax.violinplot(block_vals, positions=range(1,5), showmedians=True)
for body, col in zip(vp['bodies'], block_cols):
    body.set_facecolor(col); body.set_alpha(0.7)
vp['cmedians'].set_color('black')
ax.set_xticks(range(1,5)); ax.set_xticklabels([f'Block {b}' for b in [1,2,3,4]])
ax.set_ylabel('RMSE'); ax.set_title('RMSE Distribution by Block (Violin)', fontweight='bold')

ax2 = axes[1]
means = [v.mean() for v in block_vals]; stds = [v.std() for v in block_vals]
bars = ax2.bar(range(1,5), means, color=block_cols, alpha=0.85,
               yerr=stds, capsize=5)
ax2.set_xticks(range(1,5))
ax2.set_xticklabels([f'Block {b}\\n({BLOCK_NAMES[b].split(":")[1].strip()})'
                     for b in [1,2,3,4]], fontsize=8)
ax2.set_ylabel('Mean RMSE ± Std')
ax2.set_title('Mean RMSE by Block (v2)', fontweight='bold')
for bar, v in zip(bars, means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('inflation_v2_block_rmse.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_block_rmse.png')""")

co("""# ── Figure 3: Best per block — forecast vs actual ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axes = axes.ravel()
for i, blk in enumerate([1,2,3,4]):
    ax = axes[i]
    bm = best_per_block[blk]
    ax.plot(test_idx, y_test_raw,   color='black', lw=2,   label='Actual', zorder=3)
    ax.plot(test_idx, all_preds[bm], color=BLOCK_COLORS[blk], lw=1.6,
            label=f'{bm} (best Block {blk})', zorder=2)
    ax.plot(test_idx, all_preds['RW'], color='gray', lw=1, ls=':', alpha=0.7,
            label='Random Walk', zorder=1)
    r = np.sqrt(mean_squared_error(y_test_raw, all_preds[bm]))
    ax.set_title(f'Block {blk}: {BLOCK_NAMES[blk].split(":")[1].strip()}\\n'
                 f'Best: {bm}  |  RMSE = {r:.3f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylabel('Inflation (%)')
    if i >= 2: ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
fig.suptitle('Best-Model Forecast vs Actual — V2 (50/50 split)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('inflation_v2_best_per_block.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_best_per_block.png')""")

co("""# ── Figure 4: DM p-value heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.tril(np.ones_like(dm_pval, dtype=bool))
sns.heatmap(dm_pval_df, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn_r', vmin=0, vmax=0.2, ax=ax,
            linewidths=0.3, annot_kws={'size': 6.5},
            cbar_kws={'label': 'DM p-value', 'shrink': 0.7})
ax.set_title('Pairwise Diebold-Mariano p-values (v2)\\n'
             'Green = significant difference at p<0.05',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)
for idx in [4, 11, 14]:
    if idx <= len(model_order):
        ax.axhline(idx, color='navy', lw=2, alpha=0.7)
        ax.axvline(idx, color='navy', lw=2, alpha=0.7)
plt.tight_layout()
plt.savefig('inflation_v2_dm_heatmap.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_dm_heatmap.png')""")

co("""# ── Figure 5: v1 vs v2 DNN block comparison (qualitative summary) ─────────
dnn_models = ['LSTM','RNN','JNN','MRN','Transformer','FF-LSTM']
dnn_rmse = [np.sqrt(mean_squared_error(y_test_raw, all_preds[m])) for m in dnn_models]
# Reference best non-DNN block leaders
b1 = np.sqrt(mean_squared_error(y_test_raw, all_preds[best_per_block[1]]))
b2 = np.sqrt(mean_squared_error(y_test_raw, all_preds[best_per_block[2]]))
b3 = np.sqrt(mean_squared_error(y_test_raw, all_preds[best_per_block[3]]))

fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(dnn_models))
ax.bar(x, dnn_rmse, color=BLOCK_COLORS[4], alpha=0.85, label='DNN ensemble (v2)')
ax.axhline(b1, color=BLOCK_COLORS[1], ls='--', lw=2,
           label=f'Block 1 best ({best_per_block[1]}) RMSE={b1:.2f}')
ax.axhline(b2, color=BLOCK_COLORS[2], ls='--', lw=2,
           label=f'Block 2 best ({best_per_block[2]}) RMSE={b2:.2f}')
ax.axhline(b3, color=BLOCK_COLORS[3], ls='--', lw=2,
           label=f'Block 3 best ({best_per_block[3]}) RMSE={b3:.2f}')
ax.set_xticks(x); ax.set_xticklabels(dnn_models, fontsize=9)
ax.set_ylabel('RMSE'); ax.set_title('DNN models (v2 ensemble) vs best-of-block from other families',
                                    fontweight='bold')
for xi, v in zip(x, dnn_rmse):
    ax.text(xi, v + 0.05, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('inflation_v2_dnn_vs_blocks.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_dnn_vs_blocks.png')""")

# ─── Summary ────────────────────────────────────────────────────────────────
md("## ── SECTION 12: SUMMARY")
co("""block_best_rmse = {}
for blk in [1,2,3,4]:
    sub = results_df[results_df['Block']==blk]
    br  = sub.loc[sub['RMSE'].idxmin()]
    block_best_rmse[blk] = (br['Model'], br['RMSE'])

LINE = '=' * 70
print(LINE)
print('  SUMMARY: INFLATION FORECAST HORSE RACE — V2')
print('  Six methodological upgrades focused on giving DNNs a fair shot')
print(LINE)

print()
print('1. V2 UPGRADES ACTIVATED')
print('   ─────────────────────────────────────────────────────')
print(f'   (1) Multi-seed averaging       : 5 seeds per DNN')
print(f'   (2) Loss function              : Huber (delta=1.0)')
print(f'   (3) money_growth               : first-differenced before scaling')
print(f'   (4) vol_12m covariate          : 12-mo rolling std of inflation (lagged)')
print(f'   (5) Train/test split           : 50/50 ({SPLIT}/{n_test} obs)')
print(f'   (6) Bootstrap pre-training     : {N_BOOTSTRAP} synthetic replications')

print()
print('2. BLOCK BEST RMSE')
print('   ─────────────────────────────────────────────────────')
for blk in [1,2,3,4]:
    name_, rmse_ = block_best_rmse[blk]
    print(f'   Block {blk}  ({BLOCK_NAMES[blk]:30s}): {name_:14s}  RMSE = {rmse_:.4f}')

b1r = block_best_rmse[1][1]
b4r = block_best_rmse[4][1]
print()
print(f'   B1 → B4 improvement: {100*(b1r-b4r)/b1r:+.1f}%')

print()
print('3. WHAT TO LOOK FOR')
print('   ─────────────────────────────────────────────────────')
print('   • Compare v2 DNN ensemble RMSE to v1 single-seed RMSE')
print('   • Seed-variance table (section 9.5) quantifies single-seed noise')
print('   • If DNNs now beat Block 2/3, the v1 result was driven by the six')
print('     issues fixed here, not by an intrinsic weakness of DNNs.')
print('   • If DNNs still lose, the bottleneck is information content of the')
print('     288 (now 180) training obs, not architecture or seed luck.')
print(LINE)""")

co("""# ── Final combined figure: best per block forecasts ──────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_idx, y_test_raw, 'k-', lw=2.5, label='Actual', zorder=10)
for blk in [1,2,3,4]:
    bm = best_per_block[blk]
    r  = np.sqrt(mean_squared_error(y_test_raw, all_preds[bm]))
    ax.plot(test_idx, all_preds[bm], color=BLOCK_COLORS[blk], lw=1.5, alpha=0.85,
            label=f'B{blk} best: {bm} (RMSE={r:.2f})')
ax.set_title('V2: Best-of-block Forecasts vs Actual (50/50 split)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Inflation (%)'); ax.set_xlabel('Date')
ax.legend(fontsize=9); ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('inflation_v2_joint_best.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_joint_best.png')
print()
print('All figures saved. Notebook complete.')""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'}
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Wrote {OUT_PATH}  ({len(cells)} cells)')
