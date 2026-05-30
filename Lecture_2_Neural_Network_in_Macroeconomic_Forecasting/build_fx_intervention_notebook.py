"""Build FX_Intervention_Scenarios.ipynb — self-contained.

Application: multi-horizon DNN conditioned on reserve-drawdown paths.
Policy question: "If we burn $2bn defending the naira, what's the
expected NGN/USD path over the next 12 months?"

Audience: Central Bank FX desk / Reserves Management Committee.
"""
import nbformat as nbf

OUT_PATH = (r"G:\My Drive\Colab Notebooks\ML_WAIFEM_2026"
            r"\Lecture_2_Neural_Network_in_Macroeconomic_Forecasting"
            r"\FX_Intervention_Scenarios.ipynb")

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ─────────────────────────────────────────────────────────────────────────────
md("""# FX Intervention Scenarios — A Multi-Horizon DNN for the Reserves Management Committee
### *"If we burn \\$2bn defending the naira, what's the expected NGN/USD path over the next 12 months?"*

---

**Audience**
Central Bank FX desk, Reserves Management Committee, Treasury front office.
The framing also applies to BoG (cedi), BCEAO/CFA franc managers, the SARB
(rand) and any small-open economy where FX intervention is on the table.

**What this notebook does**

1. Generates a **synthetic monthly FX panel** with reserves, oil price,
   inflation differential, capital-flow proxy and an FX-pressure index — all
   features a real desk tracks.
2. Trains a **multi-horizon DNN** (single model, 12 output heads — one per
   month ahead) on the panel. Reserve drawdowns enter as a *conditioning input*.
3. Runs four intervention scenarios (\\$0 / \\$1bn / \\$2bn / \\$5bn defense
   over 6 months) and produces a **fan chart of NGN/USD** under each.
4. Outputs a **cost-effectiveness table** — "depreciation avoided per \\$bn burned" —
   in the language of a Reserves Management Committee paper.

**What this notebook is — and is *not***

| ✅ | ❌ |
|---|---|
| Pedagogical illustration of conditional scenario analysis for FX | A causal estimate of intervention effectiveness |
| Self-contained — runs end-to-end on any laptop in 1–3 minutes | A production tool — real deployment needs higher-frequency data and structural identification |
| Compatible with any DNN architecture for time series | Free of identifying assumptions — interventions are taken as exogenous here |

The same template plugs into **WAR (FX reaction function)** or
**event-study-style** identification once you have real intervention dates.""")

# ─── SECTION 1: imports & setup ──────────────────────────────────────────────
md("## ── SECTION 1: IMPORTS & SETUP")
co("""# !pip install -q tensorflow scikit-learn matplotlib pandas numpy seaborn

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Input, Concatenate, Dropout,
                                     RepeatVector, TimeDistributed)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})

print('TensorFlow:', tf.__version__)""")

# ─── SECTION 2: DGP ──────────────────────────────────────────────────────────
md("""## ── SECTION 2: SYNTHETIC FX DGP

We build a monthly panel for a Nigerian-style small-open economy from 2010
through 2025 (192 obs) with the following variables:

| Variable | Why it matters for FX |
|---|---|
| `fx` (NGN/USD level) | The target — log changes are what we forecast |
| `dln_fx` | Monthly log return (depreciation > 0 = naira weaker) |
| `reserves_bn` | External reserves in USD billions |
| `d_reserves` | Monthly change in reserves — proxy for *intervention* |
| `oil_brent` | Brent crude, USD/bbl — dominant export, drives reserves and the carry trade |
| `infl_diff` | Domestic minus US inflation differential (PPP pressure) |
| `cap_flow` | Portfolio-flow proxy (positive = inflows) |
| `fx_pressure` | An FX-pressure index = depreciation minus *(intervention scaled)* |

The DGP embeds three features the desk would recognise:
* **Intervention dampens depreciation contemporaneously** but with diminishing returns
  (each additional \\$bn buys less).
* **Reserve depletion below \\$30bn raises sensitivity to oil shocks** (nonlinear).
* **Positive carry from a high inflation differential attracts flows** *only* when
  reserves are above a comfort threshold.""")

co("""def generate_fx_dgp(n_obs=192, seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2010-01-01', periods=n_obs, freq='MS')
    t = np.linspace(0, 1, n_obs)

    # ── exogenous drivers ────────────────────────────────────────────────────
    oil = 70 + 25*np.sin(4*np.pi*t) + 15*np.sin(11*np.pi*t) + rng.normal(0, 6, n_obs)
    oil = np.clip(oil, 25, 140)

    infl_dom = 12 + 4*np.sin(3*np.pi*t) + rng.normal(0, 1.2, n_obs)
    infl_us  = 2  + 1*np.sin(5*np.pi*t) + rng.normal(0, 0.4, n_obs)
    infl_diff = infl_dom - infl_us

    # Capital flows: positive carry attracts flow, but only when reserves comfy
    cap_flow_base = 0.20*infl_diff + rng.normal(0, 1.0, n_obs)

    # ── pre-allocate state ───────────────────────────────────────────────────
    fx = np.zeros(n_obs); fx[0] = 150.0
    res = np.zeros(n_obs); res[0] = 45.0       # USD bn
    dln_fx = np.zeros(n_obs)
    interv = np.zeros(n_obs)                    # net USD bn sold by CBN (≥0)
    cap_flow = np.zeros(n_obs); cap_flow[0] = cap_flow_base[0]
    fx_pressure = np.zeros(n_obs)

    # ── simulate ─────────────────────────────────────────────────────────────
    for i in range(1, n_obs):
        # Reserve buffer effect: below 30bn, oil shocks bite harder
        oil_shock     = (oil[i] - oil[i-1]) / max(oil[i-1], 1.0)
        buffer_mult   = 1.0 + 2.0 * max(0.0, (30.0 - res[i-1]) / 30.0)
        oil_eff       = -0.45 * buffer_mult * oil_shock          # oil up → fx down

        # Inflation differential: persistent depreciation pressure
        infl_eff      = 0.012 * (infl_diff[i] - 5.0) / 12.0

        # Capital flows: gated by reserve comfort
        gate          = 1.0 if res[i-1] > 25.0 else 0.3
        flow          = gate * cap_flow_base[i]
        cap_flow[i]   = flow
        flow_eff      = -0.008 * flow

        # Persistence and noise
        ar_eff        = 0.30 * dln_fx[i-1]
        noise         = rng.standard_t(5) * 0.010

        # ── Authorities react: lean against persistent depreciation pressure ─
        # Reaction function (synthetic): the bank sells USD when pressure is high
        # AND reserves > 18bn (won't deplete to crisis)
        pressure_pre  = oil_eff + infl_eff + flow_eff + ar_eff
        if res[i-1] > 18 and pressure_pre > 0.005:
            interv[i] = min(0.6 + 0.5 * pressure_pre/0.02, 1.2)   # bn USD
        else:
            interv[i] = 0.0

        # Intervention dampens depreciation, with diminishing returns
        interv_eff    = -0.025 * np.tanh(interv[i] / 1.0)

        dln_fx[i]     = pressure_pre + interv_eff + noise
        fx[i]         = fx[i-1] * np.exp(dln_fx[i])
        res[i]        = max(5.0, res[i-1] - interv[i]
                            + 0.06 * (oil[i] - 60.0) / 60.0
                            + 0.20 * flow / 5.0)
        fx_pressure[i] = pressure_pre - interv_eff

    df = pd.DataFrame({
        'fx':           fx,
        'dln_fx':       dln_fx,
        'reserves_bn':  res,
        'd_reserves':   np.concatenate([[0.0], np.diff(res)]),
        'intervention': interv,
        'oil_brent':    oil,
        'infl_diff':    infl_diff,
        'cap_flow':     cap_flow,
        'fx_pressure':  fx_pressure,
    }, index=dates)
    return df

df = generate_fx_dgp()
print(f'Generated {len(df)} monthly obs from {df.index[0].date()} to {df.index[-1].date()}')
print(df.describe().round(3).to_string())""")

# ─── SECTION 3: EDA ──────────────────────────────────────────────────────────
md("## ── SECTION 3: STYLIZED FACTS")
co("""fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle('Stylized Facts of the Synthetic FX DGP', fontsize=13, fontweight='bold')

ax = axes[0, 0]
ax.plot(df.index, df['fx'], color='#D65F5F', lw=1.5)
ax.set_title('(a) NGN/USD level')
ax.set_ylabel('NGN per USD')

ax = axes[0, 1]
ax.plot(df.index, df['reserves_bn'], color='#4878CF', lw=1.5)
ax.axhline(30, color='gray', ls='--', alpha=0.6, label='Comfort threshold ($30bn)')
ax.axhline(18, color='red',  ls='--', alpha=0.6, label='Intervention floor ($18bn)')
ax.set_title('(b) External reserves'); ax.set_ylabel('USD bn'); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(df.index, 100*df['dln_fx'], color='#6ACC65', lw=0.9, alpha=0.8)
ax.axhline(0, color='black', lw=0.5)
ax.set_title('(c) Monthly log return (%)'); ax.set_ylabel('Depreciation, %')

ax = axes[1, 1]
ax.bar(df.index, df['intervention'], color='#B47CC7', width=20, alpha=0.85)
ax.set_title('(d) FX intervention (USD bn sold by CB)')
ax.set_ylabel('USD bn / month')

ax = axes[2, 0]
ax.plot(df.index, df['oil_brent'], color='#D65F5F', lw=1.2)
ax.set_title('(e) Brent oil price'); ax.set_ylabel('USD / bbl')

ax = axes[2, 1]
corr_cols = ['dln_fx','d_reserves','intervention','oil_brent','infl_diff','cap_flow']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, annot_kws={'size': 8}, ax=ax, linewidths=0.3)
ax.set_title('(f) Correlation matrix')
ax.tick_params(axis='x', rotation=35, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig('fx_eda.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: fx_eda.png')""")

# ─── SECTION 4: data prep for multi-horizon DNN ──────────────────────────────
md("""## ── SECTION 4: DATA PREP — MULTI-HORIZON SUPERVISED FORMAT

A multi-horizon DNN predicts **h-step-ahead** log returns *in one shot* for
`h = 1, 2, ..., H` (here `H = 12` months). This is the structure that
**fan-chart producers (BoE, RBNZ, Norges Bank)** prefer to recursive forecasting:
each horizon is calibrated to its own loss, errors don't compound, and
intervention scenarios are simply different input paths.

Two inputs per training example:
1. **`x_hist`** — a 24-month sequence of *past* features (8 features)
2. **`x_future_interv`** — a 12-month sequence of *future* interventions
   (USD bn per month) the scenario specifies

Output: the 12-month future path of `dln_fx`.""")

co("""FEATURE_COLS = ['dln_fx', 'd_reserves', 'oil_brent', 'infl_diff',
                'cap_flow', 'fx_pressure', 'reserves_bn', 'intervention']
TARGET_COL   = 'dln_fx'
LOOKBACK     = 24      # past months fed to encoder
HORIZON      = 12      # future months to forecast

scaler_X = StandardScaler().fit(df[FEATURE_COLS].values[:int(0.7*len(df))])
X_all    = scaler_X.transform(df[FEATURE_COLS].values)

# Intervention has its own scaler so we can manipulate it cleanly under scenarios
interv_idx_in_X = FEATURE_COLS.index('intervention')
interv_mean = scaler_X.mean_[interv_idx_in_X]
interv_scale = scaler_X.scale_[interv_idx_in_X]

def scale_interv(x_bn):     # USD bn → scaled units
    return (x_bn - interv_mean) / interv_scale

def make_multi_horizon(X, y_idx, lookback, horizon):
    \"\"\"Build (n, lookback, n_feat), (n, horizon) for one-shot forecasting.

    Also returns the FUTURE intervention path as a separate conditioning input.\"\"\"
    n = len(X) - lookback - horizon + 1
    n_feat = X.shape[1]
    Xh = np.zeros((n, lookback, n_feat))
    Xf_interv = np.zeros((n, horizon, 1))
    Y  = np.zeros((n, horizon))
    for i in range(n):
        Xh[i] = X[i:i+lookback]
        Xf_interv[i, :, 0] = X[i+lookback:i+lookback+horizon, interv_idx_in_X]
        Y[i]  = X[i+lookback:i+lookback+horizon, y_idx]
    return Xh, Xf_interv, Y

target_idx = FEATURE_COLS.index(TARGET_COL)
Xh_all, Xf_all, Y_all = make_multi_horizon(X_all, target_idx, LOOKBACK, HORIZON)
print(f'Total samples           : {len(Xh_all)}')
print(f'Encoder input shape     : {Xh_all.shape}    (past, lookback={LOOKBACK})')
print(f'Future intervention path: {Xf_all.shape}    (one channel)')
print(f'Target                  : {Y_all.shape}    (horizon={HORIZON})')

# 70/30 split (time-ordered)
n_train = int(0.7 * len(Xh_all))
Xh_tr, Xf_tr, Y_tr = Xh_all[:n_train], Xf_all[:n_train], Y_all[:n_train]
Xh_te, Xf_te, Y_te = Xh_all[n_train:], Xf_all[n_train:], Y_all[n_train:]
print(f'Train / Test            : {len(Xh_tr)} / {len(Xh_te)}')""")

# ─── SECTION 5: build encoder-decoder DNN ────────────────────────────────────
md("""## ── SECTION 5: ENCODER-DECODER DNN

Encoder–decoder with a **conditioning input** is the cleanest architecture
for scenario work:

```
              past history (24 mo, 8 feat) ─► [LSTM-64] ─► context h
                                                          │
       future intervention path (12 mo, 1 ch) ─► [LSTM-32, init=h] ─► [Dense] ─► 12-step output
```

The encoder summarises past macro state into a context vector `h`; the decoder
unrolls a 12-month forecast **conditioned on whatever future intervention path
we supply**. Swap in a different path → get a different scenario. No retraining.

We average over **3 random seeds** to dampen seed noise.""")

co("""def build_fx_seq2seq(lookback, n_feat, horizon, lr=5e-4):
    enc_in = Input(shape=(lookback, n_feat), name='past')
    enc_h  = LSTM(64, return_sequences=False, name='encoder')(enc_in)
    enc_h  = Dropout(0.2)(enc_h)

    # Repeat context across the horizon so the decoder sees it at every step
    ctx    = RepeatVector(horizon)(enc_h)

    # Decoder input = future intervention path (1 channel) concatenated with context
    dec_in = Input(shape=(horizon, 1), name='future_intervention')
    dec_x  = Concatenate(axis=-1)([ctx, dec_in])

    dec_h  = LSTM(32, return_sequences=True, name='decoder')(dec_x)
    dec_h  = Dropout(0.2)(dec_h)
    dec_h  = TimeDistributed(Dense(16, activation='relu'))(dec_h)
    out    = TimeDistributed(Dense(1))(dec_h)

    # Squeeze the trailing 1
    out    = tf.keras.layers.Reshape((horizon,))(out)
    m = Model([enc_in, dec_in], out, name='FX_Seq2Seq')
    m.compile(optimizer=Adam(lr), loss=Huber(delta=1.0))
    return m

SEEDS = [42, 43, 44]
fx_models = []
for s in SEEDS:
    tf.keras.backend.clear_session(); tf.random.set_seed(s); np.random.seed(s)
    m = build_fx_seq2seq(LOOKBACK, Xh_tr.shape[2], HORIZON)
    m.fit([Xh_tr, Xf_tr], Y_tr,
          epochs=300, batch_size=32, validation_split=0.15, verbose=0,
          callbacks=[EarlyStopping(monitor='val_loss', patience=30,
                                   restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=12, min_lr=1e-6)])
    fx_models.append(m)
    yp = m.predict([Xh_te, Xf_te], verbose=0)
    rmse_h = np.sqrt(((yp - Y_te)**2).mean(axis=0))
    print(f'  seed={s}: per-horizon RMSE (scaled dln_fx): '
          f'h1={rmse_h[0]:.3f}, h6={rmse_h[5]:.3f}, h12={rmse_h[-1]:.3f}')

print(f'{len(fx_models)}-seed ensemble ready.')""")

# ─── SECTION 6: ensemble OOS performance ─────────────────────────────────────
md("""## ── SECTION 6: OUT-OF-SAMPLE PERFORMANCE BY HORIZON

A multi-horizon model can be honest about its uncertainty: per-horizon RMSE
typically widens monotonically, and the ensemble lets us approximate **prediction
intervals** without a parametric distributional assumption (we use seed-spread
as a proxy for parameter uncertainty, and per-horizon residual std for irreducible
shock uncertainty — a common pragmatic combination).""")

co("""def ensemble_predict(models, x_enc, x_dec):
    preds = np.stack([m.predict([x_enc, x_dec], verbose=0) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)

mean_pred_te, seed_std_te = ensemble_predict(fx_models, Xh_te, Xf_te)

# Per-horizon residual std on the test set (after subtracting mean ensemble)
resid_te    = Y_te - mean_pred_te
resid_std   = resid_te.std(axis=0)
ens_h_rmse  = np.sqrt((resid_te**2).mean(axis=0))

# Total uncertainty at each horizon: combine seed-spread and residual std
total_std_h = np.sqrt(seed_std_te.mean(axis=0)**2 + resid_std**2)

perf_df = pd.DataFrame({
    'Horizon (months)':   np.arange(1, HORIZON+1),
    'RMSE (scaled)':      ens_h_rmse.round(4),
    'Seed-spread (scaled)': seed_std_te.mean(axis=0).round(4),
    'Resid std (scaled)': resid_std.round(4),
    'Total band (±1σ)':   total_std_h.round(4),
})
print('Per-horizon ensemble performance (test set):')
print(perf_df.to_string(index=False))""")

# ─── SECTION 7: scenario generator ───────────────────────────────────────────
md("""## ── SECTION 7: BUILD INTERVENTION SCENARIOS

We freeze the **encoder input** (recent history) at the *most recent* in-sample
window and unroll four different future intervention paths over the next 12 months:

| Scenario | Defense size | Path |
|---|---|---|
| **Baseline** | 0 USD bn | No CB intervention |
| **Mild defense** | 1 USD bn | 0.17 bn/month × 6 months |
| **Aggressive defense** | 2 USD bn | 0.33 bn/month × 6 months |
| **Crisis-mode defense** | 5 USD bn | 0.83 bn/month × 6 months |

Each scenario gets the **same** past-state input — the *only* thing changing is
the conditioning intervention path. This is the textbook *ceteris paribus*
the policy briefing needs.""")

co("""# Anchor on the most recent window (so the forecast emanates from "today")
x_enc_anchor = X_all[-LOOKBACK-HORIZON:-HORIZON][None, ...]   # (1, lookback, n_feat)
anchor_date  = df.index[-HORIZON-1]
forecast_idx = pd.date_range(anchor_date + pd.offsets.MonthBegin(),
                             periods=HORIZON, freq='MS')
print(f'Anchor (end of in-sample) : {anchor_date.date()}')
print(f'Forecast window           : {forecast_idx[0].date()} → {forecast_idx[-1].date()}')

SCENARIOS = {
    'Baseline ($0bn)':         (0.0, 0),
    'Mild ($1bn over 6m)':     (1.0, 6),
    'Aggressive ($2bn over 6m)': (2.0, 6),
    'Crisis ($5bn over 6m)':   (5.0, 6),
}

def build_future_interv(total_bn, n_months, horizon=HORIZON):
    \"\"\"USD bn intervention spread evenly over the first n_months of the horizon.\"\"\"
    path_bn = np.zeros(horizon)
    if n_months > 0:
        path_bn[:n_months] = total_bn / n_months
    return path_bn

scenario_paths_bn = {k: build_future_interv(b, n) for k, (b, n) in SCENARIOS.items()}
scenario_paths_sc = {k: scale_interv(p).reshape(1, -1, 1)
                     for k, p in scenario_paths_bn.items()}

print()
print('Intervention paths (USD bn per month):')
print(pd.DataFrame(scenario_paths_bn, index=forecast_idx).round(3).to_string())""")

# ─── SECTION 8: run scenarios, convert back to NGN/USD levels ────────────────
md("""## ── SECTION 8: SIMULATE NGN/USD UNDER EACH SCENARIO

Per scenario we:
1. Predict the 12-month *scaled* `dln_fx` path with all 3 seeds.
2. Un-scale to log returns; cumulate from today's level to get NGN/USD path.
3. Report the **mean path** plus a **±1σ band** derived from the combined
   seed + residual uncertainty (Section 6).""")

co("""fx_today = float(df['fx'].iloc[-HORIZON-1])
dln_mean_sc_te = X_all[:n_train, target_idx].mean()
dln_scale_sc_te = scaler_X.scale_[target_idx]
dln_mean_sc    = scaler_X.mean_[target_idx]
dln_scale_sc   = scaler_X.scale_[target_idx]

def unscale_dln(arr_sc):
    \"\"\"Un-scale the scaled target back to actual log-return units.\"\"\"
    return arr_sc * dln_scale_sc + dln_mean_sc

scenario_results = {}
for name, x_dec_sc in scenario_paths_sc.items():
    # 3-seed predictions of scaled dln_fx
    preds_sc = np.stack([m.predict([x_enc_anchor, x_dec_sc], verbose=0).ravel()
                         for m in fx_models])    # (n_seeds, horizon)
    mean_sc  = preds_sc.mean(axis=0)
    seed_sd  = preds_sc.std(axis=0)

    dln_mean   = unscale_dln(mean_sc)
    # Total uncertainty: seed dispersion (scaled) ⊕ test residual std (scaled), both un-scaled
    sd_total_sc = np.sqrt(seed_sd**2 + resid_std**2)
    dln_sd      = sd_total_sc * dln_scale_sc

    # Cumulate to FX levels
    fx_mean = fx_today * np.exp(np.cumsum(dln_mean))
    fx_lo   = fx_today * np.exp(np.cumsum(dln_mean - 1.0*dln_sd))
    fx_hi   = fx_today * np.exp(np.cumsum(dln_mean + 1.0*dln_sd))

    scenario_results[name] = {
        'dln_mean':  dln_mean,
        'fx_mean':   fx_mean,
        'fx_lo':     fx_lo,
        'fx_hi':     fx_hi,
        'total_depr_pct': float(100.0 * (fx_mean[-1] / fx_today - 1.0)),
    }

# ── Summary table ────────────────────────────────────────────────────────────
summ = pd.DataFrame({
    'Scenario':            list(scenario_results.keys()),
    '12-mo NGN/USD (mean)': [v['fx_mean'][-1] for v in scenario_results.values()],
    '12-mo depreciation %': [v['total_depr_pct'] for v in scenario_results.values()],
    'Reserves burned (USDbn)': [SCENARIOS[s][0] for s in scenario_results],
})
baseline_dep = summ.iloc[0]['12-mo depreciation %']
summ['Depreciation avoided (pp)'] = baseline_dep - summ['12-mo depreciation %']
summ['$bn burned / pp avoided']   = np.where(
    summ['Depreciation avoided (pp)'].abs() < 0.01,
    np.nan,
    summ['Reserves burned (USDbn)'] / summ['Depreciation avoided (pp)']
)
print('=' * 95)
print('  SCENARIO SUMMARY — 12-MONTH NGN/USD PATHS')
print('=' * 95)
print(summ.round(2).to_string(index=False))""")

# ─── SECTION 9: fan chart ────────────────────────────────────────────────────
md("""## ── SECTION 9: FAN CHART — NGN/USD UNDER EACH SCENARIO

This is the chart the **Reserves Management Committee** would actually look at.""")

co("""fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('FX Intervention Scenarios — Predicted NGN/USD Paths', fontsize=14, fontweight='bold')

# Recent history shown on every panel for context
hist_n = 24
hist_idx = df.index[-HORIZON-hist_n:-HORIZON]
hist_fx  = df['fx'].iloc[-HORIZON-hist_n:-HORIZON].values

colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
for ax, (name, res), col in zip(axes.ravel(), scenario_results.items(), colors):
    ax.plot(hist_idx, hist_fx, color='black', lw=1.4, label='History')
    ax.plot(forecast_idx, res['fx_mean'], color=col, lw=2.2, label=f'{name} — mean')
    ax.fill_between(forecast_idx, res['fx_lo'], res['fx_hi'], color=col, alpha=0.25,
                    label='±1σ')
    ax.axvline(forecast_idx[0], color='gray', ls=':', alpha=0.6)
    ax.axhline(fx_today, color='black', ls='--', alpha=0.4)
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('NGN per USD')
    ax.legend(fontsize=8, loc='upper left')
    ax.tick_params(axis='x', rotation=30, labelsize=8)

plt.tight_layout()
plt.savefig('fx_scenario_fanchart.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: fx_scenario_fanchart.png')""")

co("""# ── Joint panel: all four scenario means on one axis ────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hist_idx, hist_fx, color='black', lw=1.6, label='History')
for (name, res), col in zip(scenario_results.items(), colors):
    ax.plot(forecast_idx, res['fx_mean'], color=col, lw=2.2,
            label=f'{name} ({res["total_depr_pct"]:+.1f}% in 12m)')
ax.axvline(forecast_idx[0], color='gray', ls=':', alpha=0.6)
ax.axhline(fx_today, color='black', ls='--', alpha=0.4, label=f'Today: {fx_today:.0f}')
ax.set_title('NGN/USD — 12-month Forecasts Under Four Intervention Scenarios',
             fontsize=12, fontweight='bold')
ax.set_ylabel('NGN per USD'); ax.legend(fontsize=9, loc='upper left')
ax.tick_params(axis='x', rotation=30, labelsize=9)
plt.tight_layout()
plt.savefig('fx_scenario_joint.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: fx_scenario_joint.png')""")

# ─── SECTION 10: cost-effectiveness ──────────────────────────────────────────
md("""## ── SECTION 10: COST-EFFECTIVENESS — \\$bn BURNED PER PP DEPRECIATION AVOIDED

This is the single number the Reserves Management Committee actually argues
about: how many billions of reserves does it cost to prevent one percentage
point of depreciation over 12 months?

The synthetic DGP embeds diminishing returns to intervention (the `tanh`
dampener in Section 2), so we expect:

* Mild defense: relatively cheap per pp avoided
* Aggressive: rising cost per pp
* Crisis: high cost per pp — confirming the textbook *"interventions get
  more expensive the more you do them"* finding from CBN/SARB studies.""")

co("""fig, ax = plt.subplots(figsize=(11, 5.5))
non_baseline = summ.iloc[1:].copy()
bars = ax.bar(non_baseline['Scenario'], non_baseline['$bn burned / pp avoided'],
              color=colors[1:], alpha=0.85, edgecolor='black')
for bar, v in zip(bars, non_baseline['$bn burned / pp avoided']):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'${v:.2f}bn', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Cost-Effectiveness of FX Defense ($bn burned per pp depreciation avoided)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('USD bn per pp avoided')
ax.tick_params(axis='x', rotation=12, labelsize=10)
plt.tight_layout()
plt.savefig('fx_cost_effectiveness.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: fx_cost_effectiveness.png')""")

# ─── SECTION 11: briefing ────────────────────────────────────────────────────
md("## ── SECTION 11: BRIEFING-STYLE COMMENTARY")
co("""best_row = summ.iloc[1:].copy().sort_values('$bn burned / pp avoided').iloc[0]
worst_row = summ.iloc[1:].copy().sort_values('$bn burned / pp avoided', ascending=False).iloc[0]

print(f\"\"\"
─────────────────────────────────────────────────────────────────────────
  STAFF NOTE — RESERVES MANAGEMENT COMMITTEE
─────────────────────────────────────────────────────────────────────────
  Question : "What is the expected NGN/USD path over the next 12 months
              under alternative reserve-defense scenarios?"

  Method   : 3-seed ensemble of an encoder-decoder LSTM. The encoder summarises
              the past {LOOKBACK} months of FX, oil, reserves, inflation differential
              and capital-flow data; the decoder unrolls a {HORIZON}-month
              forecast conditioned on the future intervention path.

  Headline : Without intervention, the model expects NGN/USD to reach
              {scenario_results['Baseline ($0bn)']['fx_mean'][-1]:.0f}
              ({summ.iloc[0]['12-mo depreciation %']:+.1f}% from {fx_today:.0f}) over
              the next 12 months. An aggressive $2bn defense over 6 months
              reduces this to {scenario_results['Aggressive ($2bn over 6m)']['fx_mean'][-1]:.0f}
              ({summ.iloc[2]['12-mo depreciation %']:+.1f}%), avoiding
              {summ.iloc[2]['Depreciation avoided (pp)']:.1f} pp of depreciation
              at a cost of ${summ.iloc[2]['$bn burned / pp avoided']:.2f}bn per pp.

  Best     : {best_row['Scenario']} — {best_row['$bn burned / pp avoided']:.2f}bn per pp avoided
  Worst    : {worst_row['Scenario']} — {worst_row['$bn burned / pp avoided']:.2f}bn per pp avoided

  Caveats  : (i) The DNN treats intervention as exogenous to the FX shock —
              a true causal estimate requires either an IV (e.g. unanticipated
              oil shocks) or an event-study around announced intervention dates.
              (ii) The synthetic DGP embeds diminishing returns to intervention;
              real-world signs may differ. (iii) Uncertainty bands combine
              seed-spread + residual std and do NOT account for parameter
              uncertainty in the data-generating process itself.
─────────────────────────────────────────────────────────────────────────
\"\"\")""")

md("""### Extensions for a real CBN/BoG deployment

1. **Time-varying intervention paths.** Instead of an even spread, let the
   scenario specify "front-load 80% of the defense in months 1–2", which is
   how desks actually behave in a panic.
2. **Multi-currency contagion.** Add NAFEX-vs-official rate gap and parallel
   market premium as features — these are leading indicators of a forced
   devaluation.
3. **Hard reserve floor.** Add a constraint that interventions cannot push
   reserves below a critical threshold (e.g., 3 months of imports). The
   model then refuses scenarios it deems infeasible.
4. **Joint scenarios.** Combine FX intervention with an MPR change — a
   2-dimensional scenario grid is what a tight FX-monetary policy
   coordination paper would need.""")

# ──────────────────────────────────────────────────────────────────────────────
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'}
}
nbf.write(nb, OUT_PATH)
print(f'Wrote {OUT_PATH}  ({len(cells)} cells)')
