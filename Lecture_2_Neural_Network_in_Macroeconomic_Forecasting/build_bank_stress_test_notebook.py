"""Build Bank_Stress_Test_Autoencoder.ipynb — self-contained.

Application: deep autoencoder for anomaly detection on bank balance sheets.
Policy question: "Which banks look fragile under a commodity (oil) shock?"

Audience: Financial-stability department of a Central Bank, Banking
Supervision unit, or a bank's internal IFRS 9 / ICAAP team.
"""
import nbformat as nbf

OUT_PATH = (r"G:\My Drive\Colab Notebooks\ML_WAIFEM_2026"
            r"\Lecture_2_Neural_Network_in_Macroeconomic_Forecasting"
            r"\Bank_Stress_Test_Autoencoder.ipynb")

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ─────────────────────────────────────────────────────────────────────────────
md("""# Bank Stress Test via Deep Autoencoder
### *"Which banks look fragile under a commodity (oil) shock?"*

---

**Audience**
Central Bank Financial Stability department, Banking Supervision desks (CBN, BoG,
SARB), IFRS 9 / ICAAP teams inside commercial banks, and macro-prudential units
in finance ministries.

**Why deep autoencoders for stress testing?**

Traditional stress tests use **structural models** of credit risk that require
strong parametric assumptions on default correlations, PD curves and LGD. An
autoencoder takes a complementary, **data-driven** approach:

1. Train an autoencoder to reconstruct the **normal-period** balance-sheet
   profile of each bank — the network learns the *typical correlation structure*
   between CAR, NPL, ROA, LDR, FX exposure and sectoral concentration.
2. Apply a macro shock (here: oil –40%) propagated through bank-specific
   sectoral exposures.
3. **Banks whose post-shock profile has a high reconstruction error are
   "anomalous"** — they look unlike anything the network saw in normal times.
   Those are the candidates for closer supervisory attention.

This is the same idea used by:
* The ECB's market-data anomaly detection for early-warning indicators
* Several major banks' internal model validation pipelines
* Fraud / financial crime detection (where this technique originated)

**What this notebook is — and is *not***

| ✅ | ❌ |
|---|---|
| A flexible *screening* tool that flags banks for deeper investigation | A replacement for a full bottom-up stress test |
| Robust to nonlinear interactions a parametric model would miss | A causal model of failure — high error is a *signal*, not a verdict |
| Reproducible end-to-end on a laptop | Production-ready — real deployment needs supervisory data, full panel |""")

# ─── SECTION 1: imports ──────────────────────────────────────────────────────
md("## ── SECTION 1: IMPORTS & SETUP")
co("""# !pip install -q tensorflow scikit-learn matplotlib pandas numpy seaborn

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})

print('TensorFlow:', tf.__version__)""")

# ─── SECTION 2: bank panel DGP ───────────────────────────────────────────────
md("""## ── SECTION 2: SYNTHETIC BANK PANEL

We simulate **24 banks × 48 months** of bank-month observations spanning two
regimes:

| Period | Months | Description |
|---|---|---|
| **Normal** | 1 – 36 | All banks operate under benign macro conditions |
| **Stressed (held out)** | 37 – 48 | A commodity shock hits oil-exposed banks |

Each bank has 10 quarterly-reported balance-sheet ratios. We assign each bank
a **business model archetype** that drives its baseline ratios and its
sensitivity to oil:

| Archetype | n banks | Key feature |
|---|---|---|
| Retail-heavy | 8 | Low FX exposure, low oil sensitivity |
| Corporate / Oil & Gas | 6 | High concentration in oil & gas — **most exposed** to commodity shocks |
| Trade finance | 4 | High FX exposure, moderate oil sensitivity |
| Government securities | 4 | Heavy treasury holdings, low oil sensitivity |
| Universal large | 2 | Diversified, mid sensitivity |

The autoencoder is trained ONLY on the *normal* period. The stressed months are
held out so the model has never seen the post-shock balance sheets. Banks that
the model can't reconstruct = fragility signal.""")

co("""# ── Bank archetypes ─────────────────────────────────────────────────────────
ARCHETYPES = {
    'Retail':       {'n': 8, 'car_mu':16.0, 'npl_mu':4.0,  'roa_mu':2.0, 'ldr_mu':70.0,
                     'fx_mu':10.0, 'oil_conc_mu': 5.0, 'gov_conc_mu':15.0,
                     'oil_beta':0.20, 'fx_beta':0.20},
    'Oil-Gas':      {'n': 6, 'car_mu':15.0, 'npl_mu':6.0,  'roa_mu':2.5, 'ldr_mu':80.0,
                     'fx_mu':35.0, 'oil_conc_mu':45.0, 'gov_conc_mu': 5.0,
                     'oil_beta':1.20, 'fx_beta':0.55},
    'TradeFinance': {'n': 4, 'car_mu':14.0, 'npl_mu':5.0,  'roa_mu':2.2, 'ldr_mu':75.0,
                     'fx_mu':55.0, 'oil_conc_mu':10.0, 'gov_conc_mu':10.0,
                     'oil_beta':0.45, 'fx_beta':0.95},
    'GovSec':       {'n': 4, 'car_mu':18.0, 'npl_mu':3.0,  'roa_mu':1.8, 'ldr_mu':55.0,
                     'fx_mu':12.0, 'oil_conc_mu': 3.0, 'gov_conc_mu':55.0,
                     'oil_beta':0.10, 'fx_beta':0.20},
    'UniversalLg':  {'n': 2, 'car_mu':15.5, 'npl_mu':5.0,  'roa_mu':2.1, 'ldr_mu':72.0,
                     'fx_mu':25.0, 'oil_conc_mu':18.0, 'gov_conc_mu':20.0,
                     'oil_beta':0.55, 'fx_beta':0.40},
}

# Build bank list with archetype tags
banks = []
bank_id = 0
for arch, p in ARCHETYPES.items():
    for k in range(p['n']):
        banks.append({'bank_id': f'B{bank_id:02d}', 'archetype': arch,
                      'oil_beta': p['oil_beta'], 'fx_beta': p['fx_beta']})
        bank_id += 1
banks_df = pd.DataFrame(banks)
print(f'Total banks: {len(banks_df)}')
print(banks_df.head(10))""")

co("""def generate_bank_panel(banks_df, n_months_normal=36, n_months_stress=12, seed=SEED):
    rng = np.random.default_rng(seed)
    n_months = n_months_normal + n_months_stress
    dates    = pd.date_range('2022-01-01', periods=n_months, freq='MS')

    # ── Macro environment ───────────────────────────────────────────────────
    # Oil drops 40% gradually in the stress period
    oil = np.ones(n_months) * 80.0
    for i in range(n_months_normal, n_months):
        frac = (i - n_months_normal + 1) / n_months_stress
        oil[i] = 80.0 * (1 - 0.40 * frac)
    oil = oil + rng.normal(0, 2.0, n_months)

    # FX (NGN/USD) — gradual depreciation, sharper in stress
    fx = 750.0 + np.linspace(0, 100, n_months) + rng.normal(0, 8, n_months)
    fx[n_months_normal:] += np.linspace(0, 250, n_months_stress)

    macro = pd.DataFrame({'oil': oil, 'fx': fx,
                          'oil_shock': (oil - 80.0) / 80.0,
                          'fx_shock':  (fx - 750.0) / 750.0}, index=dates)

    rows = []
    for _, b in banks_df.iterrows():
        arch_p = ARCHETYPES[b['archetype']]
        # Baseline ratios for THIS bank — small heterogeneity around archetype
        car_base = arch_p['car_mu']    + rng.normal(0, 0.5)
        npl_base = arch_p['npl_mu']    + rng.normal(0, 0.4)
        roa_base = arch_p['roa_mu']    + rng.normal(0, 0.2)
        ldr_base = arch_p['ldr_mu']    + rng.normal(0, 2.0)
        fx_exp_base = arch_p['fx_mu']  + rng.normal(0, 2.0)
        oil_conc_base = arch_p['oil_conc_mu'] + rng.normal(0, 1.5)
        gov_conc_base = arch_p['gov_conc_mu'] + rng.normal(0, 1.5)

        # Persistent latent factors (bank-specific quality)
        lat = np.zeros(n_months)
        for t in range(1, n_months):
            lat[t] = 0.90 * lat[t-1] + rng.normal(0, 0.10)

        for t, dt in enumerate(dates):
            oil_shk = macro['oil_shock'].iloc[t]
            fx_shk  = macro['fx_shock'].iloc[t]
            # Capital ratio falls under oil shock (proportional to oil_beta)
            car  = car_base - 4.0 * b['oil_beta'] * max(0, -oil_shk)  + 0.5*lat[t] + rng.normal(0, 0.20)
            # NPL rises with the shock
            npl  = npl_base + 8.0 * b['oil_beta'] * max(0, -oil_shk) - 0.3*lat[t] + rng.normal(0, 0.25)
            # Profitability hit
            roa  = roa_base - 2.5 * b['oil_beta'] * max(0, -oil_shk) + 0.4*lat[t] + rng.normal(0, 0.10)
            # LDR — slight rise
            ldr  = ldr_base + 5.0 * b['oil_beta'] * max(0, -oil_shk) + rng.normal(0, 0.6)
            # FX exposure (NGN equivalent): rises with FX shock proportional to fx_beta
            fx_exp = fx_exp_base * (1 + 0.30 * b['fx_beta'] * fx_shk) + rng.normal(0, 0.5)
            oil_conc = oil_conc_base + rng.normal(0, 0.5)
            gov_conc = gov_conc_base + rng.normal(0, 0.5)
            # Liquidity coverage ratio (LCR) — drops in stress
            lcr  = 140.0 - 20.0 * b['oil_beta'] * max(0, -oil_shk) + rng.normal(0, 3.0)
            # Cost-to-income (CIR) — rises in stress
            cir  = 55.0  + 15.0 * b['oil_beta'] * max(0, -oil_shk) + rng.normal(0, 1.5)
            # Net interest margin (NIM) — narrows
            nim  = 6.0   - 1.5 * b['oil_beta'] * max(0, -oil_shk) + rng.normal(0, 0.15)

            rows.append({
                'date': dt, 'bank_id': b['bank_id'], 'archetype': b['archetype'],
                'period': ('stress' if t >= n_months_normal else 'normal'),
                'CAR': car, 'NPL': npl, 'ROA': roa, 'LDR': ldr,
                'FX_exposure': fx_exp, 'oil_conc': oil_conc, 'gov_conc': gov_conc,
                'LCR': lcr, 'CIR': cir, 'NIM': nim,
                'oil': macro['oil'].iloc[t], 'fx': macro['fx'].iloc[t],
            })
    panel = pd.DataFrame(rows)
    return panel, macro

panel, macro = generate_bank_panel(banks_df)
RATIO_COLS = ['CAR','NPL','ROA','LDR','FX_exposure','oil_conc','gov_conc','LCR','CIR','NIM']
print(f'Panel shape: {panel.shape}  ({panel["bank_id"].nunique()} banks × {panel["date"].nunique()} months)')
print()
print('Sample (first bank, 4 normal + 2 stress months):')
print(panel[panel.bank_id=='B00'].iloc[[0,1,2,3,36,37]][['date','period']+RATIO_COLS].round(2).to_string(index=False))""")

# ─── SECTION 3: EDA ──────────────────────────────────────────────────────────
md("## ── SECTION 3: STYLIZED FACTS")
co("""fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Bank Panel — Stylized Facts', fontsize=13, fontweight='bold')

# (a) Macro environment
ax = axes[0, 0]
ax.plot(macro.index, macro['oil'], color='#D65F5F', lw=1.6, label='Oil ($/bbl)')
ax.axvspan(macro.index[36], macro.index[-1], color='red', alpha=0.10, label='Stress')
ax.set_title('(a) Oil price (commodity shock at month 37)'); ax.legend(fontsize=8)
ax.set_ylabel('USD / bbl')

# (b) Cross-sectional CAR distribution over time
ax = axes[0, 1]
piv_car = panel.pivot_table(index='date', columns='bank_id', values='CAR')
ax.plot(piv_car.index, piv_car.values, color='steelblue', lw=0.6, alpha=0.4)
ax.plot(piv_car.index, piv_car.median(axis=1), color='black', lw=2.2, label='Median CAR')
ax.axvspan(macro.index[36], macro.index[-1], color='red', alpha=0.10)
ax.axhline(10, color='red', ls='--', alpha=0.6, label='Reg minimum 10%')
ax.set_title('(b) CAR by bank — note divergence in stress'); ax.legend(fontsize=8)
ax.set_ylabel('CAR (%)')

# (c) NPL distribution
ax = axes[1, 0]
piv_npl = panel.pivot_table(index='date', columns='bank_id', values='NPL')
ax.plot(piv_npl.index, piv_npl.values, color='#D65F5F', lw=0.6, alpha=0.4)
ax.plot(piv_npl.index, piv_npl.median(axis=1), color='black', lw=2.2, label='Median NPL')
ax.axvspan(macro.index[36], macro.index[-1], color='red', alpha=0.10)
ax.set_title('(c) NPL by bank'); ax.legend(fontsize=8); ax.set_ylabel('NPL (%)')

# (d) Archetype averages — CAR
ax = axes[1, 1]
for arch in ARCHETYPES:
    sub = panel[panel.archetype==arch].groupby('date')['CAR'].mean()
    ax.plot(sub.index, sub.values, lw=1.7, label=arch)
ax.axvspan(macro.index[36], macro.index[-1], color='red', alpha=0.10)
ax.set_title('(d) Average CAR by archetype'); ax.legend(fontsize=7)
ax.set_ylabel('CAR (%)')

# (e) Correlation in normal period
ax = axes[2, 0]
corr = panel[panel.period=='normal'][RATIO_COLS].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1,
            annot_kws={'size':7}, ax=ax, linewidths=0.3)
ax.set_title('(e) Ratio correlation in normal period')
ax.tick_params(axis='x', rotation=45, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)

# (f) Correlation in stress period
ax = axes[2, 1]
corr_s = panel[panel.period=='stress'][RATIO_COLS].corr()
sns.heatmap(corr_s, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1,
            annot_kws={'size':7}, ax=ax, linewidths=0.3)
ax.set_title('(f) Ratio correlation in stress period\\n(notice the shift — autoencoder picks this up)')
ax.tick_params(axis='x', rotation=45, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig('bank_stress_eda.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: bank_stress_eda.png')""")

# ─── SECTION 4: build & train autoencoder ────────────────────────────────────
md("""## ── SECTION 4: BUILD THE DEEP AUTOENCODER

A **deep undercomplete autoencoder**:

```
input (10 ratios) ─► [16] ─► [8] ─► bottleneck (4) ─► [8] ─► [16] ─► output (10 ratios)
```

The bottleneck of 4 forces the network to learn a *compressed* representation
of the normal-period correlation structure. Anything that doesn't fit that
structure (a bank with simultaneously plunging CAR, surging NPL and shrinking
NIM — a stressed bank) will be reconstructed poorly.

Training set: **normal period only** (3 × 36 = 1248 bank-months × 10 ratios,
with a 80/20 train/val split).""")

co("""# ── Scaling: fit ONLY on the normal period to avoid contamination ──────────
normal_mask = (panel['period'] == 'normal')
X_normal    = panel.loc[normal_mask, RATIO_COLS].values
X_stress    = panel.loc[~normal_mask, RATIO_COLS].values

scaler = StandardScaler().fit(X_normal)
Xn_sc  = scaler.transform(X_normal)
Xs_sc  = scaler.transform(X_stress)
print(f'Train (normal) : {Xn_sc.shape}')
print(f'Held out (stress): {Xs_sc.shape}')""")

co("""def build_autoencoder(input_dim, bottleneck=4):
    inp = Input(shape=(input_dim,), name='ratios')
    x   = Dense(16, activation='relu')(inp)
    x   = BatchNormalization()(x)
    x   = Dropout(0.10)(x)
    x   = Dense(8,  activation='relu')(x)
    z   = Dense(bottleneck, activation='linear', name='bottleneck')(x)
    x   = Dense(8,  activation='relu')(z)
    x   = Dense(16, activation='relu')(x)
    out = Dense(input_dim, activation='linear', name='reconstruction')(x)
    ae = Model(inp, out, name='AE')
    ae.compile(optimizer=Adam(1e-3), loss='mse')
    return ae

tf.keras.backend.clear_session()
tf.random.set_seed(SEED); np.random.seed(SEED)

ae = build_autoencoder(len(RATIO_COLS), bottleneck=4)
ae.summary()""")

co("""# ── Train on normal period ──────────────────────────────────────────────────
hist = ae.fit(Xn_sc, Xn_sc,
              epochs=300, batch_size=64, validation_split=0.2, verbose=0,
              shuffle=True,
              callbacks=[EarlyStopping(monitor='val_loss', patience=25,
                                       restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=10, min_lr=1e-5)])

print(f'Final train loss     : {hist.history["loss"][-1]:.5f}')
print(f'Final validation loss: {hist.history["val_loss"][-1]:.5f}')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist.history['loss'],     label='Train MSE')
ax.plot(hist.history['val_loss'], label='Val MSE')
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (log)')
ax.set_title('Autoencoder Training Loss'); ax.legend()
plt.tight_layout()
plt.savefig('bank_stress_ae_loss.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: bank_stress_ae_loss.png')""")

# ─── SECTION 5: reconstruction error scoring ─────────────────────────────────
md("""## ── SECTION 5: COMPUTE RECONSTRUCTION ERROR

For every bank-month we compute the **squared reconstruction error** averaged
across the 10 ratios. This is the bank-month *anomaly score*. We compare:

* **Normal-period error** (baseline noise floor)
* **Stress-period error** (the signal we want)
* **Per-bank error** during stress (the fragility ranking — *who* looks
  unusual, not just whether the system as a whole is stressed)""")

co("""def recon_error(model, X):
    \"\"\"Per-row mean squared reconstruction error.\"\"\"
    Xhat = model.predict(X, verbose=0)
    return ((X - Xhat)**2).mean(axis=1)

# All bank-months in original order, with the period tag
panel = panel.sort_values(['bank_id','date']).reset_index(drop=True)
X_all  = scaler.transform(panel[RATIO_COLS].values)
panel['recon_err'] = recon_error(ae, X_all)

# Threshold = 95th percentile of normal-period errors
thresh = np.quantile(panel.loc[panel.period=='normal', 'recon_err'], 0.95)
print(f'95th-percentile threshold (normal period): {thresh:.4f}')
print()
print(panel.groupby('period')['recon_err']
      .describe()[['count','mean','50%','max']]
      .round(4))""")

# ─── SECTION 6: time-series view ────────────────────────────────────────────
md("""## ── SECTION 6: TIME-SERIES VIEW — WHEN DID THE SYSTEM TRIGGER?

We plot the *median bank-month error across all 24 banks* through time. The
expected pattern: noisy and below threshold in the normal period; rising and
crossing the threshold during the oil shock.""")

co("""err_ts = panel.groupby('date')['recon_err'].agg(['median','mean','quantile'])
err_q90 = panel.groupby('date')['recon_err'].quantile(0.9)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(err_ts.index, err_ts['median'], color='#4878CF', lw=1.6, label='Cross-section median')
ax.plot(err_ts.index, err_q90,          color='#D65F5F', lw=1.6, label='Cross-section 90th pctile')
ax.axhline(thresh, color='black', ls='--', lw=1.2, label=f'Normal-period 95% threshold ({thresh:.3f})')
ax.axvspan(macro.index[36], macro.index[-1], color='red', alpha=0.10, label='Oil shock period')
ax.set_yscale('log')
ax.set_title('Cross-Section Reconstruction Error Over Time\\n'
             '(autoencoder trained on normal period only)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Reconstruction error (log)'); ax.legend(fontsize=9)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('bank_stress_error_timeseries.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: bank_stress_error_timeseries.png')""")

# ─── SECTION 7: per-bank fragility ranking ───────────────────────────────────
md("""## ── SECTION 7: PER-BANK FRAGILITY RANKING

The headline output for Banking Supervision: which banks crossed the
reconstruction-error threshold during the stress period, and by how much?""")

co("""stress_err = (panel[panel.period=='stress']
              .groupby(['bank_id','archetype'])['recon_err']
              .mean()
              .reset_index()
              .sort_values('recon_err', ascending=False))

stress_err['z_vs_normal'] = (
    (stress_err['recon_err'] - panel.loc[panel.period=='normal','recon_err'].mean())
    / panel.loc[panel.period=='normal','recon_err'].std()
)
stress_err['flag'] = np.where(stress_err['recon_err'] > thresh, 'FLAG', '–')

print('PER-BANK STRESS-PERIOD ANOMALY SCORE')
print('=' * 70)
print(stress_err.round(3).to_string(index=False))""")

co("""# ── Visualize ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
arch_colors = {'Retail':'#4878CF','Oil-Gas':'#D65F5F','TradeFinance':'#6ACC65',
               'GovSec':'#B47CC7','UniversalLg':'#8C8C8C'}
colors = stress_err['archetype'].map(arch_colors).values

bars = ax.barh(stress_err['bank_id'], stress_err['recon_err'],
               color=colors, alpha=0.85, edgecolor='black')
ax.axvline(thresh, color='black', ls='--', lw=1.5,
           label=f'95% threshold from normal period ({thresh:.3f})')
ax.set_xlabel('Mean reconstruction error during stress period')
ax.set_title('Bank Fragility Ranking — Commodity (Oil –40%) Shock\\n'
             'Banks above threshold = candidates for supervisory deep-dive',
             fontsize=12, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=a) for a, c in arch_colors.items()]
legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Above threshold'))
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

for bar, v, flag in zip(bars, stress_err['recon_err'], stress_err['flag']):
    ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
            f'{v:.3f} {"★" if flag=="FLAG" else ""}',
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig('bank_stress_fragility_ranking.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: bank_stress_fragility_ranking.png')""")

# ─── SECTION 8: per-ratio decomposition ──────────────────────────────────────
md("""## ── SECTION 8: WHY IS THIS BANK ANOMALOUS? PER-RATIO DECOMPOSITION

A supervisor flagging a bank needs to explain *which* indicator is driving the
anomaly. We decompose the bank's reconstruction error per ratio for the most
fragile bank — pinpointing the single ratio (CAR? NPL? FX exposure?) that
diverges most from the historical correlation structure.""")

co("""# Identify top-flagged bank
top_bank = stress_err.iloc[0]['bank_id']
print(f'Most anomalous bank during stress: {top_bank} '
      f'({stress_err.iloc[0]["archetype"]})')

# Reconstruct that bank's stress-period observations
mask_bank   = (panel.bank_id == top_bank) & (panel.period == 'stress')
X_bank_sc   = scaler.transform(panel.loc[mask_bank, RATIO_COLS].values)
X_bank_rec  = ae.predict(X_bank_sc, verbose=0)
ratio_err   = (X_bank_sc - X_bank_rec)**2          # (n_obs, n_ratios)
ratio_err_mean = ratio_err.mean(axis=0)

decomp = pd.Series(ratio_err_mean, index=RATIO_COLS,
                   name=f'Avg squared recon-error contribution ({top_bank}, stress)')
decomp_sorted = decomp.sort_values(ascending=False)

print()
print(f'Per-ratio reconstruction-error contribution for {top_bank}:')
print('-' * 60)
print(decomp_sorted.round(4).to_string())

# Visualize — original vs reconstructed for top bank
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
ax.bar(decomp_sorted.index, decomp_sorted.values,
       color=['#D65F5F' if v > decomp.mean() else '#4878CF' for v in decomp_sorted.values],
       alpha=0.85, edgecolor='black')
ax.set_title(f'(a) Per-ratio anomaly contribution — {top_bank}', fontweight='bold')
ax.set_ylabel('Mean squared reconstruction error')
ax.tick_params(axis='x', rotation=35, labelsize=9)

# (b) Trajectory of the most problematic ratio: original vs reconstruction
top_ratio_idx = int(np.argmax(ratio_err_mean))
top_ratio_name = RATIO_COLS[top_ratio_idx]
months_stress = pd.to_datetime(panel.loc[mask_bank, 'date'].values)

ax = axes[1]
ax.plot(months_stress, X_bank_sc[:, top_ratio_idx], 'ko-', lw=2.0, label='Observed (scaled)')
ax.plot(months_stress, X_bank_rec[:, top_ratio_idx], 's--', color='#D65F5F', lw=1.5,
        label='AE reconstruction (what the model "expects")')
ax.fill_between(months_stress, X_bank_sc[:, top_ratio_idx], X_bank_rec[:, top_ratio_idx],
                color='#D65F5F', alpha=0.20, label='Anomaly gap')
ax.set_title(f'(b) {top_ratio_name}: observed vs reconstructed for {top_bank}',
             fontweight='bold')
ax.set_ylabel(f'{top_ratio_name} (z-scored)')
ax.legend(fontsize=9)
ax.tick_params(axis='x', rotation=30, labelsize=8)

plt.tight_layout()
plt.savefig('bank_stress_per_ratio_decomp.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: bank_stress_per_ratio_decomp.png')""")

# ─── SECTION 9: archetype-level heatmap ─────────────────────────────────────
md("""## ── SECTION 9: ARCHETYPE-LEVEL HEATMAP — WHICH BUSINESS MODELS ARE AT RISK?

A macro-prudential view: which **business models** as a class are showing
elevated anomaly scores during the shock?""")

co("""arch_err = (panel
            .groupby(['archetype', 'date'])['recon_err']
            .mean()
            .reset_index())
arch_piv = arch_err.pivot(index='archetype', columns='date', values='recon_err')

fig, ax = plt.subplots(figsize=(14, 4.5))
sns.heatmap(arch_piv, cmap='RdYlGn_r',
            cbar_kws={'label': 'Mean reconstruction error'}, ax=ax,
            linewidths=0.1, vmin=arch_piv.values.min(), vmax=arch_piv.values.max())
ax.axvline(36, color='black', lw=2.0)
ax.text(36.5, -0.4, '← Oil shock begins', fontsize=9, fontweight='bold')
ax.set_title('Anomaly Score by Business Model Over Time\\n'
             '(red = high reconstruction error = looks unusual vs normal-period structure)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('')

# Format x-axis with sparser dates
xt = np.arange(0, len(arch_piv.columns), 6)
ax.set_xticks(xt); ax.set_xticklabels([arch_piv.columns[i].strftime('%Y-%m') for i in xt],
                                      rotation=30, fontsize=8)
plt.tight_layout()
plt.savefig('bank_stress_archetype_heatmap.png', bbox_inches='tight', dpi=120); plt.show()
print('Saved: bank_stress_archetype_heatmap.png')""")

# ─── SECTION 10: briefing ───────────────────────────────────────────────────
md("## ── SECTION 10: BRIEFING-STYLE COMMENTARY")
co("""n_flagged = (stress_err['flag'] == 'FLAG').sum()
flagged_banks = stress_err.loc[stress_err.flag=='FLAG', 'bank_id'].tolist()
top_arch = (stress_err
            .groupby('archetype')['recon_err']
            .mean()
            .sort_values(ascending=False))
worst_arch = top_arch.index[0]

print(f\"\"\"
─────────────────────────────────────────────────────────────────────────
  STAFF NOTE — FINANCIAL STABILITY DEPARTMENT
─────────────────────────────────────────────────────────────────────────
  Question : "Which banks would look fragile under a 40% oil-price decline?"

  Method   : Deep autoencoder (10→16→8→4→8→16→10) trained on 36 months of
              pre-shock balance-sheet ratios across {panel.bank_id.nunique()}
              banks. A bank-month is flagged anomalous if its reconstruction
              error exceeds the 95th percentile of the normal-period error
              distribution ({thresh:.3f}).

  Headline : {n_flagged} of {panel.bank_id.nunique()} banks crossed the threshold
              during the simulated stress period. Most fragile business model:
              '{worst_arch}' (avg anomaly score {top_arch.iloc[0]:.3f}, vs system
              normal-period mean of {panel.loc[panel.period=="normal","recon_err"].mean():.3f}).

  Flagged  : {", ".join(flagged_banks) if flagged_banks else "(none)"}

  Top bank : {top_bank} — driven mostly by anomalous behavior in '{decomp_sorted.index[0]}'
              and '{decomp_sorted.index[1]}'.

  Caveats  : (i) High reconstruction error is a SIGNAL, not a verdict — it flags
              banks for closer supervisory examination, not for resolution.
              (ii) The autoencoder learns the normal-period CORRELATION structure;
              a bank with deteriorating fundamentals that *still respects that
              correlation* will not be flagged. Pair this tool with a structural
              early-warning model that monitors levels.
              (iii) The synthetic DGP assumes oil-sensitivity by archetype; in
              practice, granular sector-of-loan-portfolio data is needed.
─────────────────────────────────────────────────────────────────────────
\"\"\")""")

md("""### Extensions for a real Financial Stability deployment

1. **Variational autoencoder (VAE)** — gives a *probabilistic* anomaly score
   (negative log-likelihood under the learned density), which is more
   defensible in a supervisory letter than a raw MSE threshold.
2. **Conditional autoencoder** — condition on macro state so the same balance
   sheet is judged differently under different oil prices.
3. **Cross-bank graph autoencoder** — adds interbank exposure as edges, picking
   up contagion paths.
4. **Counterfactual stress generation** — instead of waiting for stress, use
   a generative model (GAN or normalizing flow) to *synthesise* stressed
   scenarios consistent with the bank panel's correlation structure, then run
   the autoencoder on them. This is what the BoE's RAMSI-style models do.
5. **Pair with a survival model** — for banks repeatedly flagged across
   scenarios, feed the anomaly score into a logistic model with historical
   bank-failure data to translate the score into a default probability.

Together with the inflation counterfactual (Section 13 of
`Inflation_Forecast_Comparison_v2.ipynb`) and FX intervention scenarios
(`FX_Intervention_Scenarios.ipynb`), this gives a **three-application
template** for NN-based policy and supervisory analytics suitable for any
emerging-market central bank or ministry.""")

# ──────────────────────────────────────────────────────────────────────────────
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'}
}
nbf.write(nb, OUT_PATH)
print(f'Wrote {OUT_PATH}  ({len(cells)} cells)')
