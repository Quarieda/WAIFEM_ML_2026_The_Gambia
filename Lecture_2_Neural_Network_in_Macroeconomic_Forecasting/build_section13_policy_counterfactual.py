"""Append Section 13: Policy Counterfactual Demo to Inflation_Forecast_Comparison_v2.ipynb.

The added section:
  13.1  Why a Central Bank cares about NN counterfactuals
  13.2  Train a single counterfactual-ready LSTM (fast, single seed)
  13.3  Construct factual vs counterfactual MPR (interest_rate) paths
  13.4  Re-predict inflation under each path
  13.5  Gradient-based driver attribution (poor-man's SHAP via tf.GradientTape)
  13.6  Policymaker-ready chart pack + briefing-style commentary
"""
import nbformat as nbf

NB_PATH = (r"G:\My Drive\Colab Notebooks\ML_WAIFEM_2026"
           r"\Lecture_2_Neural_Network_in_Macroeconomic_Forecasting"
           r"\Inflation_Forecast_Comparison_v2.ipynb")

nb = nbf.read(NB_PATH, as_version=4)
new = []
md = lambda s: new.append(nbf.v4.new_markdown_cell(s))
co = lambda s: new.append(nbf.v4.new_code_cell(s))

# ──────────────────────────────────────────────────────────────────────────────
md("""---
# SECTION 13 — Policy Counterfactual Demo
## *"What if the MPR had stayed at 14% instead of being raised to 27.5%?"*

This is the language a **Monetary Policy Committee briefing** speaks. The horse-race
above tells us *which* model forecasts inflation best out-of-sample; it does **not**
tell us what would have happened under a different policy. This section bridges that gap.

**What you'll see here**

1. A deliberately simple counterfactual: hold the policy rate (`interest_rate`)
   at its *pre-tightening* level for the entire test window, leaving every other
   exogenous driver (oil, FX, fiscal, money) at its realised path.
2. The trained DNN re-scores inflation under that counterfactual path.
3. A gradient-based attribution (a transparent stand-in for SHAP) decomposes the
   gap between factual and counterfactual inflation into per-feature contributions.
4. A briefing-pack chart in the style an MPC member would actually see.

**What this demo is — and is *not***

| | |
|---|---|
| ✅ A pedagogical illustration of how NNs can be plugged into scenario work | ❌ A causal identification — we are interpreting a *flexible conditional expectation*, not a structural impulse-response |
| ✅ Cheap and reproducible — runs in seconds on any laptop | ❌ Production-grade — a real CBN/BoG/BoE briefing would use a held-out, regime-stratified backtest |
| ✅ Compatible with any model in the horse race | ❌ Free of identifying assumptions — we still need the policy rate to be (approximately) exogenous to current-period shocks |

A real-world deployment would *combine* this with a structural model (e.g. a small
semi-structural DSGE for narrative, an NN for the data-fit residual) — the
NN excels at picking up nonlinear pass-through that the structural model
mis-specifies.""")

# ─── 13.1 Build a counterfactual-ready single-seed LSTM ───────────────────────
md("""## 13.1 Train a single counterfactual-ready LSTM

The horse-race code averaged 5 seeds and only kept the mean prediction (good for
forecasting; bad for counterfactual analysis because we lose the model object).
We retrain **one** LSTM with `seed=42`, reusing every preprocessing artifact
from the horse race (`scaler_X`, `scaler_y`, `X_seq_train`, `y_seq_train`, `SEQ_LEN`,
`N_FEAT`, `FEATURE_COLS`, `df`). This takes ~20–60 seconds.""")

co("""# Lightweight refit — same architecture as build_lstm(), single seed
tf.keras.backend.clear_session()
tf.random.set_seed(42); np.random.seed(42)

cf_model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, N_FEAT), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1),
], name='CF_LSTM')
cf_model.compile(optimizer=Adam(3e-4), loss=Huber(delta=1.0))

cf_model.fit(
    X_seq_train, y_seq_train,
    epochs=200, batch_size=64,
    validation_split=0.15, verbose=0,
    callbacks=[EarlyStopping(monitor='val_loss', patience=25,
                             restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=10, min_lr=1e-6)],
)
factual_pred_sc = cf_model.predict(X_seq_test, verbose=0).ravel()
factual_pred    = inverse_scale_y(factual_pred_sc)
rmse_cf = float(np.sqrt(mean_squared_error(y_test_raw, factual_pred)))
print(f'Counterfactual-ready LSTM trained.  Test RMSE = {rmse_cf:.4f}')""")

# ─── 13.2 Build counterfactual feature paths ──────────────────────────────────
md("""## 13.2 Construct the counterfactual policy path

We freeze the policy rate (`interest_rate`) at its **median value across the
first 36 months of the test window** (a stylised "no-tightening" stance) for
the *entire* test window, leaving every other driver at its realised path.

The choice of "freeze at the early-test median" is generic — in a real exercise
you would freeze at a Taylor-rule baseline, or at the rate observed at a specific
policy meeting, etc.""")

co("""# Identify the column index of the policy rate in FEATURE_COLS
POLICY_COL_NAME = 'interest_rate'
policy_idx = FEATURE_COLS.index(POLICY_COL_NAME)

# Counterfactual: hold policy rate at its early-test median for whole horizon
early_test_window = 36
cf_policy_level   = float(np.median(df[POLICY_COL_NAME].values[SPLIT:SPLIT+early_test_window]))

X_cf_full = X_full.copy()
X_cf_full[SPLIT:, policy_idx] = cf_policy_level     # counterfactual stance

# Rebuild scaled sequences under the counterfactual
X_cf_sc      = scaler_X.transform(X_cf_full)
X_cf_seq, _  = make_sequences(X_cf_sc, y_sc_all, SEQ_LEN)
X_cf_seq_test = X_cf_seq[SPLIT_SEQ:]

cf_pred_sc = cf_model.predict(X_cf_seq_test, verbose=0).ravel()
cf_pred    = inverse_scale_y(cf_pred_sc)

print(f'Factual policy rate over test window  : mean={df[POLICY_COL_NAME].values[SPLIT:].mean():.2f}, '
      f'max={df[POLICY_COL_NAME].values[SPLIT:].max():.2f}')
print(f'Counterfactual policy rate (constant)  : {cf_policy_level:.2f}')
print()
print('Predicted inflation paths over test window:')
print(f'  Factual    (model under realised MPR) : mean={factual_pred.mean():6.2f}, max={factual_pred.max():6.2f}')
print(f'  Counterfac (MPR held at {cf_policy_level:.2f})       : mean={cf_pred.mean():6.2f}, max={cf_pred.max():6.2f}')
print(f'  Average gap (factual - counterfactual): {(factual_pred - cf_pred).mean():+.3f} pp')""")

# ─── 13.3 Gradient-based driver attribution (poor-man's SHAP) ─────────────────
md("""## 13.3 Gradient-based driver attribution

A Monetary Policy Committee member will not accept *"the model says inflation
would have been 3 percentage points higher"* without a decomposition: how much
came from the policy rate, how much from oil, how much from FX?

We compute **input-gradient attributions** — the sensitivity of each predicted
inflation point to each input feature at each lag — and aggregate them per
feature. This is a transparent, dependency-free stand-in for SHAP values:

> *attribution<sub>j</sub>(t) ≈ Σ<sub>ℓ</sub>  (∂ŷ<sub>t</sub> / ∂x<sub>j,t-ℓ</sub>) · (x<sub>j,t-ℓ</sub><sup>factual</sup> − x<sub>j,t-ℓ</sub><sup>counterfactual</sup>)*

This is the **first-order Taylor approximation** of the prediction gap. It is
exact when the model is locally linear (which it is, near any single input) and
a reasonable proxy when the gap is not too large.""")

co("""@tf.function
def _model_jacobian(x_batch):
    \"\"\"Return d(yhat)/d(x_batch) for a batch of sequences.

    Output shape: (batch, seq_len, n_feat) — sensitivity per (timestep, feature).\"\"\"
    x = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        yhat = cf_model(x, training=False)
    grads = tape.gradient(yhat, x)
    return grads

# Sensitivities at the FACTUAL inputs (Taylor expansion point)
grads_factual = _model_jacobian(X_seq_test).numpy()        # (n_test_seq, SEQ_LEN, N_FEAT)

# Δx in standardised units (same units the model sees)
delta_X = X_seq_test - X_cf_seq_test                       # factual - counterfactual

# Per-feature contribution to the inflation gap, summed across the SEQ_LEN lags
attrib_sc = (grads_factual * delta_X).sum(axis=1)          # (n_test_seq, N_FEAT)

# Convert the SCALED-target contributions back to inflation %-points
scale_y = scaler_y.scale_[0]
attrib_pp = attrib_sc * scale_y                            # (n_test_seq, N_FEAT)

# Tabulate average contribution per feature (test-window mean)
attrib_mean = pd.Series(attrib_pp.mean(axis=0), index=FEATURE_COLS,
                        name='Avg. contribution to (factual - counterfactual), pp')
attrib_mean_sorted = attrib_mean.reindex(attrib_mean.abs().sort_values(ascending=False).index)

print('Driver attribution — average %-point contribution to the inflation gap')
print('(positive = pushed factual inflation ABOVE the counterfactual)')
print('-' * 70)
print(attrib_mean_sorted.round(3).to_string())
print('-' * 70)
print(f'Sum of attributions       : {attrib_mean.sum():+.3f} pp')
print(f'Actual mean gap           : {(factual_pred - cf_pred).mean():+.3f} pp')
print('  (gap residual = nonlinearity not captured by first-order attribution)')""")

# ─── 13.4 Policymaker chart pack ──────────────────────────────────────────────
md("""## 13.4 Policymaker chart pack

Four panels designed for a one-page MPC briefing:

1. **Top-left** — Factual vs Counterfactual MPR paths
2. **Top-right** — Predicted inflation under each path (this is the headline number)
3. **Bottom-left** — Inflation *gap* (factual − counterfactual) over time, with horizons annotated
4. **Bottom-right** — Driver attribution bar chart (what's pushing the gap)""")

co("""fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Policy Counterfactual: MPR held at {:.1f}% vs realised path'
             .format(cf_policy_level), fontsize=14, fontweight='bold')

# ── (a) MPR paths ────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(test_idx, df[POLICY_COL_NAME].values[SPLIT:], color='#D65F5F', lw=2.0,
        label=f'Realised MPR (factual)')
ax.axhline(cf_policy_level, color='#4878CF', lw=2.0, ls='--',
           label=f'Counterfactual MPR (= {cf_policy_level:.1f}%)')
ax.fill_between(test_idx, cf_policy_level, df[POLICY_COL_NAME].values[SPLIT:],
                where=(df[POLICY_COL_NAME].values[SPLIT:] > cf_policy_level),
                color='#D65F5F', alpha=0.15, label='Tightening relative to CF')
ax.set_title('(a) Policy rate paths', fontsize=11, fontweight='bold')
ax.set_ylabel('MPR (%)'); ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30, labelsize=8)

# ── (b) Inflation paths ──────────────────────────────────────────────────────
ax = axes[0, 1]
test_seq_idx = test_idx[SEQ_LEN:][:len(factual_pred)] if len(test_idx) >= len(factual_pred) + SEQ_LEN else test_idx[-len(factual_pred):]
ax.plot(test_seq_idx, y_test_raw[SEQ_LEN:][:len(factual_pred)],
        color='black', lw=1.5, alpha=0.6, label='Realised inflation')
ax.plot(test_seq_idx, factual_pred,  color='#D65F5F', lw=2.0,
        label='Model — factual MPR')
ax.plot(test_seq_idx, cf_pred,       color='#4878CF', lw=2.0,
        label='Model — counterfactual MPR')
ax.fill_between(test_seq_idx, factual_pred, cf_pred,
                where=(factual_pred > cf_pred), color='#D65F5F', alpha=0.18,
                label='Inflation suppressed by tightening')
ax.fill_between(test_seq_idx, factual_pred, cf_pred,
                where=(factual_pred <= cf_pred), color='#4878CF', alpha=0.18,
                label='Inflation higher under factual')
ax.set_title('(b) Predicted inflation under each policy path',
             fontsize=11, fontweight='bold')
ax.set_ylabel('Inflation (%)'); ax.legend(fontsize=7, loc='upper right')
ax.tick_params(axis='x', rotation=30, labelsize=8)

# ── (c) Inflation gap over time ──────────────────────────────────────────────
ax = axes[1, 0]
gap = factual_pred - cf_pred
ax.plot(test_seq_idx, gap, color='#6ACC65', lw=2.0)
ax.axhline(0, color='black', lw=0.8, ls=':')
ax.fill_between(test_seq_idx, 0, gap, where=(gap < 0),
                color='#4878CF', alpha=0.25, label='Tightening lowers inflation')
ax.fill_between(test_seq_idx, 0, gap, where=(gap >= 0),
                color='#D65F5F', alpha=0.25, label='Tightening raises inflation (unusual)')
ax.set_title('(c) Inflation gap: factual minus counterfactual',
             fontsize=11, fontweight='bold')
ax.set_ylabel('Percentage points'); ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30, labelsize=8)

# ── (d) Driver attribution bar chart ─────────────────────────────────────────
ax = axes[1, 1]
attrib_plot = attrib_mean_sorted
colors_attr = ['#D65F5F' if v > 0 else '#4878CF' for v in attrib_plot.values]
bars = ax.barh(attrib_plot.index, attrib_plot.values, color=colors_attr, alpha=0.85)
ax.axvline(0, color='black', lw=0.8)
for bar, v in zip(bars, attrib_plot.values):
    ax.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
            f'{v:+.2f}', va='center', ha=('left' if v >= 0 else 'right'),
            fontsize=8)
ax.set_title('(d) Driver attribution (avg pp contribution to gap)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Percentage points')

plt.tight_layout()
plt.savefig('inflation_v2_policy_counterfactual.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: inflation_v2_policy_counterfactual.png')""")

# ─── 13.5 Briefing commentary ─────────────────────────────────────────────────
md("""## 13.5 Briefing-style commentary

For the WAIFEM lecture, it helps to show participants *exactly* how the model
output gets translated into the language of a Monetary Policy Committee paper.""")

co("""# ── Auto-generated MPC-style briefing paragraph ─────────────────────────────
gap_mean = float(gap.mean())
gap_max  = float(np.max(np.abs(gap)))
top_driver = attrib_mean_sorted.index[0]
top_val    = float(attrib_mean_sorted.iloc[0])
second_driver = attrib_mean_sorted.index[1]
second_val    = float(attrib_mean_sorted.iloc[1])

briefing = f\"\"\"
─────────────────────────────────────────────────────────────────────────
  STAFF NOTE — MONETARY POLICY COUNTERFACTUAL
─────────────────────────────────────────────────────────────────────────
  Question : "What would inflation have been if the policy rate had been held
              at {cf_policy_level:.1f}% throughout the test window?"

  Method   : Deep LSTM trained on the full sample through {df.index[SPLIT-1].date()},
              re-scored on the test window with the interest-rate input replaced by
              a constant {cf_policy_level:.1f}% and all other drivers at their realised
              paths.

  Headline : The realised tightening lowered predicted inflation by an average of
              {-gap_mean:+.2f} percentage points relative to the counterfactual,
              with a peak monthly impact of {gap_max:.2f} percentage points.

  Drivers  : Of the average gap, the largest contributor was '{top_driver}'
              ({top_val:+.2f} pp), followed by '{second_driver}' ({second_val:+.2f} pp).

  Caveats  : This is a CONDITIONAL EXPECTATION exercise. Identification of a
              causal monetary-policy effect would require either (a) a structural
              model with a Taylor-rule reaction function, or (b) an
              instrumental-variable strategy isolating exogenous policy
              surprises. The figure above should therefore be read as the
              model's best estimate of inflation under the assumed policy path,
              not as a structural impulse-response.
─────────────────────────────────────────────────────────────────────────\"\"\"
print(briefing)""")

md("""### Extensions worth showing to a CBN/BoG/MoF audience

1. **Fan-chart version.** Repeat 13.2–13.4 across multiple seeds (or with MC-dropout
   left enabled at inference) to get an *uncertainty band* around the
   counterfactual path. This converts the demo into a Bank-of-England-style
   density chart.
2. **Multiple counterfactual paths.** Instead of one constant rate, try
   `cf_policy_level ∈ {14, 18, 22, 27.5}` and overlay the inflation paths.
   This is the chart a Governor wants for a policy-tightening cost-benefit
   discussion.
3. **Combine with structural narrative.** Run the same counterfactual through
   a small DSGE and show where the NN and DSGE disagree. The gap is where
   non-linear / non-Gaussian dynamics live — exactly what the NN is supposed
   to capture.
4. **State-dependent policy.** Allow the counterfactual rate to depend on the
   inflation regime indicator (`df['regime']`), turning a pure scenario into a
   simple model-implied reaction function.

The notebooks `FX_Intervention_Scenarios.ipynb` and
`Bank_Stress_Test_Autoencoder.ipynb` apply the *same* template to two other
policy questions — FX defense and financial-stability stress testing.""")

# ──────────────────────────────────────────────────────────────────────────────
nb['cells'].extend(new)
nbf.write(nb, NB_PATH)
print(f'Appended {len(new)} cells.  Notebook now has {len(nb["cells"])} cells.')
print(f'Saved: {NB_PATH}')
