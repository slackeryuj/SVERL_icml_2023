
# Methodology — SHAP‑Informed Two‑Stage Portfolio Learning Pipeline (v3)

> **Scope.** **Stage 1** predicts next‑day returns per asset with XGBoost and exports SHAP‑based factor attributions. **Stage 2** uses a monthly‑step reinforcement‑learning agent (PPO) to allocate portfolio weights, consuming predictions and SHAP features (plus optional technicals). The procedure is **universe‑agnostic** and applies to sector ETFs and to equity universes such as DIA constituents.

---

## 1. Notation and Data Construction

Let \(\mathcal{U}=\{1,\dots,N\}\) be the asset universe. For asset \(i\) on day \(t\): adjusted close \(P_{t,i}\), return \(r_{t,i}=\frac{P_{t,i}}{P_{t-1,i}}-1\). Feature families:

- **ETF‑specific technicals** \(\mathcal{S}=\{\text{SMA}_5,\text{EMA}_{12},\text{RSI}_7,\text{MACD},\text{ATR},\text{Vol}_5,\text{Mom}_3,\text{LagRet}_1,\text{LagRet}_2,\text{LagRet}_3\}\).
- **Macro factors** with lags \(\mathcal{F}=\{\text{Mkt-RF}_{t-\ell},\text{SMB}_{t-\ell},\text{HML}_{t-\ell},\text{RMW}_{t-\ell},\text{CMA}_{t-\ell}\}\) for \(\ell\in\{0,1,2,3\}\) (+ VIX if available).

All features are observable at \(t\). Missing **features** may be forward‑filled; labels are **never** forward‑filled.

---

## 2. Rolling Time Splits

- **Stage 1:** Train 12y / Valid 1y / Test 1y; annual retraining. Purge \(h=1\) rows from Train and Valid to prevent look‑ahead.  
- **Stage 2:** Train 7y / Valid 1y / Predict 1y; windows advance by 252 trading days. One rebalance per 21 trading days.

---

## 3. Stage 1 — XGBoost Forecasters + Unified SHAP Gate

### 3.1 Labels
Next‑day return \(y_{t+1,i}=r_{t+1,i}\). (Optional risk‑adjusted label \(\tilde y_{t+1,i}=r_{t+1,i}/\widehat\sigma_{t,i}\), \(w{=}20\) days, is disabled in the current run.)

### 3.2 Gate: stable feature selection across assets
Fit “gate” XGBoost models per asset on a base window and compute mean‑absolute SHAP to rank features. Select at least **MIN_SUFFIXES = 6** ETF suffix groups and at most **MAX_GLOBAL_FAMILIES = 4** macro families, filling to **TOP_N_GENERIC = 10** total groups. Cache the manifest (years, horizon, universe).

### 3.3 **XGBoost Overview** 
XGBoost (eXtreme Gradient Boosting) learns an **additive ensemble of regression trees** by **second‑order** gradient boosting with explicit regularization. For data \(D=\{(x_i,y_i)\}_{i=1}^n\), the model after \(M\) trees is
\[
\hat y_i^{(M)}=\sum_{m=1}^M f_m(x_i), \qquad f_m\in\mathcal{F},
\]
where \(\mathcal{F}\) is the class of CART trees. The training objective (squared‑error loss) adds regularization on tree complexity and leaf weights:
\[
\mathcal{L}^{(M)}=\sum_{i=1}^n \tfrac{1}{2}\big(y_i-\hat y_i^{(M)}\big)^2 \;+\; \sum_{m=1}^M \Omega(f_m),\qquad 
\Omega(f)=\gamma T+\tfrac{\lambda}{2}\sum_{j=1}^{T} w_j^2 \;(+\,\alpha\!\sum_j|w_j|\ \text{if L1, here } \alpha{=}0),
\]
with \(T\) leaves and leaf output \(w_j\).

**Functional boosting step.** Given the current prediction \(\hat y^{(M-1)}\), add one tree \(f_M\) by minimizing a **second‑order** Taylor expansion of \(\mathcal{L}\) around \(\hat y^{(M-1)}\):
\[
\tilde{\mathcal{L}}^{(M)}=\sum_{i=1}^n \big[g_i f_M(x_i)+\tfrac{1}{2}h_i f_M(x_i)^2\big]+\Omega(f_M),
\]
where \(g_i=\partial_{\hat y}\ell(y_i,\hat y)\), \(h_i=\partial^2_{\hat y}\ell(y_i,\hat y)\) at \(\hat y=\hat y_i^{(M-1)}\). For squared error: \(g_i=\hat y_i^{(M-1)}-y_i\), \(h_i=1\). For a tree with leaf index sets \(\{I_j\}\), the **optimal leaf weight** and **node split gain** are
\[
w_j^\star=-\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda},\qquad
\text{Gain}=\tfrac{1}{2}\!\left(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right)-\gamma,
\]
with \(G=\sum g_i,\ H=\sum h_i\) over the node; left/right similarly. A split is accepted if \(\text{Gain}>0\) and child nodes satisfy `min_child_weight` (lower bound on \(H\)).

**Shrinkage and stochasticity.** The new tree is added with learning rate \(\eta\in(0,1]\):
\[
\hat y^{(M)}=\hat y^{(M-1)}+\eta\, f_M(x).
\]
`subsample` (rows) and `colsample_bytree` (columns) inject randomness to reduce variance.

**Regularization links to our hyperparameters.** We use `reg:squarederror` with \(\lambda\in\{1,2\}\), `max_depth∈{3,4,5}`, `min_child_weight=1`, `subsample∈{0.8,0.9}`, `colsample_bytree∈{0.8,1.0}`, `n_estimators∈{600,1000}`, `learning_rate∈{0.05,0.10}`; gate models use a lighter setting (`n_estimators=512`, `max_depth=4`, `learning_rate=0.08`).

### 3.4 **Training → Validation selection → Test prediction**
For each asset \(i\) and year \(Y\):
1. **Assemble** \((X^{\text{train}},y^{\text{train}})\), \((X^{\text{valid}},y^{\text{valid}})\), \((X^{\text{test}},y^{\text{test}})\) with the **fixed** gate manifest; purge the last \(h=1\) rows from Train/Valid.
2. **Preprocess**: fit `SimpleImputer(median)` and `StandardScaler` on Train only; transform Valid/Test.
3. **Grid search & early stopping** over the hyperparameter set above. Select the model that **maximizes validation Spearman IC** (rank correlation between \(\hat y\) and \(y\)); RMSE is the tie‑breaker. Spearman IC is
\[
\rho=\operatorname{corr}_\text{rank}\big(\hat y^{\text{valid}}, y^{\text{valid}}\big).
\]
4. **Predict** on Test (only dates with realizable labels): \(\hat r_{t+1,i}=f_\theta(X_{t,i})\). Create cross‑sectional percentile **score** \(s_{t,i}\in[0,1]\).

### 3.5 **From forecasts to SHAP features**
We compute SHAP values \(\phi_{t,i,j}\) with **TreeSHAP**, which evaluates the exact **Shapley values** for tree ensembles:
\[
\phi_j(x)=\!\!\sum_{S\subseteq\mathcal{F}\setminus\{j\}}\!\!\frac{|S|!\,(M-|S|-1)!}{M!}\Big(f_x(S\cup\{j\})-f_x(S)\Big),\quad 
f(x)=\phi_0+\sum_{j=1}^M\phi_j(x),
\]
with \(\phi_0=\mathbb{E}[f(X)]\). We aggregate SHAP into factor families and ETF‑suffix groups:
\[
\phi_{t,i,g}=\sum_{j\in\text{family}(g)}\phi_{t,i,j},\qquad
\phi_{t,i,s}=\sum_{j\in\text{suffix}(s)}\phi_{t,i,j},
\]
and build **relative (demeaned) SHAP** for cross‑sectional comparability,
\[
\phi^{\text{REL}}_{t,i,g}=\phi_{t,i,g}-\tfrac{1}{N}\sum_{k=1}^N\phi_{t,k,g}.
\]
The Stage‑1 output table per date contains: `Actual_Return_i`, `Predicted_Return_i`, `Predicted_Score_i`, `SHAP_*` (family/suffix and REL variants).

---

## 4. Stage 2 — PPO Allocation Agent (Monthly Macro‑Step)

The agent observes daily features but **acts monthly**. Within the month, weights drift self‑financing.

### 4.1 MDP and accounting
State \(s_t=\operatorname{vec}([x_{t-L+1},\ldots,x_t])\) with \(L=21\). Action \(a_t\in[-1,1]^N\) is mapped to weights using temperature‑scaled squashing \(z_i=\tanh(u_i/\tau)\) with \(\tau=0.4\) and a **baseline‑relative** softmax tilt \(w\propto b\odot \exp(z)\) projected to long‑only bounds/budgets. Weights drift inside the month:
\[
\tilde w_{t+1}= \frac{w_t\odot (1+r_{t+1})}{1+w_t^\top r_{t+1}}.
\]
Monthly reward at boundary \(m\):
\[
R_m=\Big(\prod_{d\in m}(1+w_d^\top r_d)-1\Big)\ -\ \lambda_{\text{risk}}\operatorname{CVaR}_\alpha \ -\ \lambda_T\,\tfrac{1}{2}\|w_m-w_{m^-}\|_1\ -\ \lambda_H\sum_i w_{m,i}^2\ -\ c\,\tfrac{1}{2}\|w_m-w_{m^-}\|_1.
\]

### 4.2 Observation design
Per asset \(i\): `Predicted_Return_i`; (optional) **SHAP channels** (DIR/CONF/RESID, REL SHAP) built from §3.5; and (optional) technicals (indicator set or `momentum_basic`). Non‑SHAP columns are standardized; SHAP columns are robust‑scaled to \([q_5,q_{95}]\) and amplified by \(\gamma_{\text{shap}}=1.3\); a SHAP‑only dropout with \(p=0.10\) is applied during training. Rewards are normalized via `VecNormalize` (observations are not).

### 4.3 **Proximal Policy Optimization Overview**
Proximal Policy Optimization (denoted as "PPO") is an **on‑policy** actor–critic method that performs multiple SGD steps on a **clipped** policy‑gradient objective to limit destructive updates. Let \(\pi_\theta(a\mid s)\) be the policy, \(V_\theta(s)\) the value function, and \(r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}\) the probability ratio. With empirical advantages \(\hat A_t\), the **clipped surrogate** is
\[
\mathcal{L}_{\text{clip}}(\theta)=\mathbb{E}\Big[\min\big(r_t(\theta)\,\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big].
\]
We add a value‑loss and an entropy bonus:
\[
\mathcal{L}_{\text{PPO}}(\theta)=\mathcal{L}_{\text{clip}}(\theta)\ -\ c_v\,\mathbb{E}\big[\,(V_\theta(s_t)-\hat R_t)^2\,\big]\ +\ c_H\,\mathbb{E}\big[\mathcal{H}(\pi_\theta(\cdot\mid s_t))\big],
\]
and **maximize** \(\mathcal{L}_{\text{PPO}}\).

**Advantage estimation (GAE).** With temporal‑difference residual \(\delta_t=r_t+\gamma V_\theta(s_{t+1})-V_\theta(s_t)\), the Generalized Advantage Estimator is
\[
\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\qquad \hat R_t=\hat A_t+V_\theta(s_t).
\]
In our setup, the reward \(r_t\) is **zero on intra‑monthly days** and equals \(R_m\) at the monthly boundary (the wrapper aggregates daily steps).

### 4.4 **Train PPO Agent** 
For each window and configuration:
1. **Rollouts.** Collect `n_steps∈{64,96,128}` daily transitions with the monthly‑reward wrapper. Store \((s_t,a_t,r_t,\log\pi_{\theta_{\text{old}}},V_{\theta_{\text{old}}})\).
2. **Return/advantage targets.** Compute normalized returns \(\hat R_t\) and GAE advantages \(\hat A_t\) (we **standardize** \(\hat A_t\) per batch).
3. **SGD epochs.** For `n_epochs=5`, do minibatch updates to **maximize** \(\mathcal{L}_{\text{PPO}}\) using Adam with learning rate in \(\{3\!\times\!10^{-4},2\!\times\!10^{-4},10^{-4}\}\), clip \(\epsilon\in\{0.1,0.2\}\), `vf_coef∈{0.5,0.7,1.0}`, `ent_coef∈{0.002,0.005,0.01}`, `max_grad_norm∈{0.3,0.5}`. Update \(\theta_{\text{old}}\leftarrow\theta\).
4. **Validation & early stopping.** Evaluate monthly reward on the **validation slice**; keep the best model, stop after two non‑improvements. We enforce a **feasibility screen**: deviation from equal‑weight in \([0.25,0.70]\) and average monthly turnover \(\le 0.60\).
5. **Prediction.** Freeze the best policy and run month‑by‑month on the OOS slice to log `weights.csv`, `wealth.csv`, and metrics.

### 4.5 Baseline portfolio \(b\) (stabilization)
Scores \(s_{t,i}\) (predictions if present, otherwise indicator composite) are z‑scored cross‑sectionally; apply softmax with temperature \(k=10\), clip to \([0.02,0.22]\), renormalize, and EMA‑smooth with 3‑month half‑life. Actions generate **relative tilts** around \(b\), improving stability and diversification.

---

## 5. Reproducibility and Anti‑Leakage

Purged splits; Train‑only transformers; observable‑only features; monthly macro‑step accounting; cached gate manifest and VecNormalize state; deterministic seeds with per‑window offsets.

---

## 6. Parameterization (current run)

**Stage 1 (gate & forecasters).** Gate: `n_estimators=512`, `max_depth=4`, `learning_rate=0.08`, `tree_method="hist"`, `device="cuda"`, `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=1.0`, `reg_lambda=1.0`. Selection budgets: **TOP_N_GENERIC = 10**, **MIN_SUFFIXES = 6**, **MAX_GLOBAL_FAMILIES = 4**. Per‑asset grid: `max_depth∈{3,4,5}`, `learning_rate∈{0.05,0.10}`, `n_estimators∈{600,1000}`, `subsample∈{0.8,0.9}`, `colsample_bytree∈{0.8,1.0}`, `reg_lambda∈{1.0,2.0}`, seeds `{42,202}`; select by validation IC (tie‑break RMSE). Splits: 12y/1y/1y; horizon \(h=1\).

**Stage 2 (PPO).** Windows 7y/1y/1y; `lookback_period=21`, `rebalance_period=21`; action temperature \(\tau=0.4\); long‑only \([0,1]\) with budgets \(L=1.0, S=0.0\). Reward: Mean − \(\lambda_{\text{risk}}\)\(\mathrm{CVaR}_{0.05}\) − \(0.003\cdot\mathrm{TO}\) − \(0.04\cdot\mathrm{HHI}\) − \(5\times10^{-4}\cdot\mathrm{TO}\); warm‑up 12 months; \(\lambda_{\text{risk}}\in\{0.10,0.25,0.50,1.00\}\) tuned. PPO search: learning rate \(\in\{3\!\times\!10^{-4},2\!\times\!10^{-4},10^{-4}\}\), `n_steps∈{64,96,128}`, `batch_size∈{32,64}`, `gamma∈{0.96,0.985,0.995}`, `gae_lambda∈{0.90,0.95,0.98}`, `clip_range∈{0.1,0.2}`, `vf_coef∈{0.5,0.7,1.0}`, `ent_coef∈{0.002,0.005,0.01}`, `max_grad_norm∈{0.3,0.5}`; incremental training up to 12,000 steps with patience=2; 30 iterations with seeded offsets.

---

### Summary Equation Block

- Monthly return: \(\displaystyle R_m=\prod_{d\in m}(1+w_d^\top r_d)-1\).  
- Self‑financing drift: \(\displaystyle \tilde w_{t+1}=\frac{w_t\odot(1+r_{t+1})}{1+w_t^\top r_{t+1}}\).  
- CVaR: \(\displaystyle \operatorname{CVaR}_\alpha=\mathbb{E}[\mathcal{L}\mid \mathcal{L}\ge \operatorname{VaR}_\alpha],\ \mathcal{L}=-R\).  
- PPO objective: \(\displaystyle \max_\theta\ \mathbb{E}\big[ \min(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)\big] - c_v(V_\theta-\hat R)^2 + c_H\mathcal{H}(\pi_\theta)\).  
- GAE: \(\displaystyle \hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\quad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\).  
- XGBoost split gain: \(\displaystyle \tfrac{1}{2}\!\left(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right)-\gamma\).  
- SHAP additivity: \(\displaystyle f(x)=\phi_0+\sum_j\phi_j(x)\).

---

**Artifacts.**  
- **Stage 1 → Stage 2 feed:** `stage1_predictions_with_shap_<UNIVERSE>.csv` with `Date, ETF, Actual_Return, Predicted_Return, Predicted_Score, SHAP_*` (+ REL variants).  
- **Stage 1 cache:** `stage1_unified_gate_metrics.csv`, per‑asset/year `{xgb_regressor.json, scaler.pkl, imputer.pkl, meta.json}`.  
- **Stage 2:** `stage2_iterations/iteration_*/window_*/{weights,wealth,training_validation_log}.csv`, `feature_list.{txt,csv}`, `best_params.json`, PPO model, VecNormalize state.
