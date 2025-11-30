## 3. Methodology

We develop a hierarchical **signal‑to‑policy** framework for dynamic, risk‑aware portfolio optimization over a universe of liquid ETFs (in the experiments, \(N = 10\) sector ETFs). The framework is explicitly two‑stage:

1. **Stage 1 (supervised prediction):** per‑ETF forward return forecasts with XGBoost, using a unified feature set of technical indicators and Fama–French–style factor/macro series, selected via a SHAP‑based gate.
2. **Stage 2 (policy optimization):** a PPO actor–critic agent that consumes a compressed signal layer (Stage 1 predictions, Fama–French factors, and selected technical/momentum signals) and outputs portfolio weights under a sequential mean–CVaR objective with trading and concentration penalties.

Sections 3.1–3.5 formalize the theoretical foundations and both stages, and then integrate them into a coherent signal‑to‑policy mapping.

---

### 3.1 Theoretical Foundations: Fama–French Models and Risk–Return Optimization

#### 3.1.1 Multi‑factor asset pricing and Fama–French models

Let \(r_{i,t+1}\) denote the **simple return** of asset \(i\) between dates \(t\) and \(t+1\), and let \(r_{f,t+1}\) be the corresponding risk‑free rate. In a generic \(K\)‑factor model, excess returns follow
$$
r_{i,t+1} - r_{f,t+1}
\;=\; \alpha_i \;+\; \beta_i^\top f_{t+1} \;+\; \varepsilon_{i,t+1},
$$
where:

- \(f_{t+1} \in \mathbb{R}^K\) is the vector of factor realizations (e.g., market, size, value, momentum),
- \(\beta_i \in \mathbb{R}^K\) are factor loadings,
- \(\varepsilon_{i,t+1}\) is idiosyncratic noise with
  \(\mathbb{E}[\varepsilon_{i,t+1} \mid f_{t+1}] = 0\).

Under no‑arbitrage and correct model specification, \(\alpha_i \approx 0\), and expected excess returns satisfy
$$
\mathbb{E}\big[r_{i,t+1} - r_{f,t+1}\big]
\;=\;
\beta_i^\top \lambda,
$$
where \(\lambda \in \mathbb{R}^K\) are factor risk premia.

The Fama–French five‑factor model specifies
$$
f_{t+1} =\begin{bmatrix}
\text{MKT}_{t+1} \\
\text{SMB}_{t+1} \\
\text{HML}_{t+1} \\
\text{RMW}_{t+1} \\
\text{CMA}_{t+1}
\end{bmatrix},
$$
where:

- \(\text{MKT}_{t+1}\) = market excess return,
- \(\text{SMB}_{t+1}\) = size factor (small minus big),
- \(\text{HML}_{t+1}\) = value factor (high minus low book‑to‑market),
- \(\text{RMW}_{t+1}\) = profitability factor (robust minus weak),
- \(\text{CMA}_{t+1}\) = investment factor (conservative minus aggressive).

An extended specification often adds a **momentum factor** \(\text{UMD}_{t+1}\) and other macro variables (e.g., volatility or term‑structure proxies). In practice, these are implemented as factor‑mimicking portfolios whose returns can be used directly as features or to construct factor‑tilted allocations. :contentReference[oaicite:0]{index=0}  

For a portfolio with weights \(w \in \mathbb{R}^N\), the portfolio excess return is
$$
r_{p,t+1} - r_{f,t+1}=w^\top \big( r_{t+1} - r_{f,t+1}\mathbf{1} \big)=\alpha_p + \beta_p^\top f_{t+1} + \varepsilon_{p,t+1},
$$
with \(\beta_p = \sum_i w_i \beta_i\) and \(\alpha_p = \sum_i w_i \alpha_i\). This representation makes factor exposures explicit and motivates:

- our use of Fama–French factors as predictors in Stage 1, and  
- factor‑grouped SHAP attributions as an interpretable decomposition of model forecasts and portfolio behavior.

#### 3.1.2 Mean–variance portfolio optimization

Let \(r_{t+1} \in \mathbb{R}^N\) denote the **vector of asset returns** over \([t, t+1]\). Assume \(\mu = \mathbb{E}[r_{t+1}]\) and \(\Sigma = \operatorname{Var}(r_{t+1})\) are known or estimated. A mean–variance investor with risk‑aversion parameter \(\gamma > 0\) solves:
$$
\begin{aligned}
\max_{w \in \mathbb{R}^N} \quad &U_{\text{MV}}(w)=w^\top \mu-\frac{\gamma}{2}\, w^\top \Sigma w, \\
\text{subject to} \quad &
\mathbf{1}^\top w = 1, \\
& w \in \mathcal{W},
\end{aligned}
$$
where \(\mathcal{W}\) encodes additional constraints (long‑only, leverage limits, factor‑exposure bounds, etc.). :contentReference[oaicite:1]{index=1}  

In the unconstrained case \(\mathcal{W} = \mathbb{R}^N\), the first‑order condition yields
$$
\mu - \gamma \Sigma w^* = \lambda \mathbf{1},
$$
for some scalar \(\lambda\) enforcing the budget constraint. This yields the classical mean–variance efficient portfolio (after accounting for the risk‑free asset). However, total variance penalizes upside and downside volatility symmetrically and does not explicitly target tail risk, motivating alternative risk measures.

#### 3.1.3 Mean–VaR portfolio optimization

Define portfolio loss
$$
L_{t+1}(w) = - w^\top r_{t+1}.
$$
For confidence level \(\alpha \in (0,1)\), the **Value‑at‑Risk (VaR)** at level \(\alpha\) is
$$
\text{VaR}_\alpha(L)=\inf\left\{ \ell \in \mathbb{R} : \mathbb{P}(L \le \ell) \ge \alpha \right\}.
$$
VaR is the smallest loss threshold not exceeded with probability at least \(\alpha\). A mean–VaR investor solves
$$
\begin{aligned}
\max_{w \in \mathbb{R}^N} \quad &U_{\text{MVaR}}(w)=w^\top \mu-\lambda_{\text{var}}\, \text{VaR}_\alpha\big(L(w)\big), \\
\text{subject to} \quad &
\mathbf{1}^\top w = 1,\quad w \in \mathcal{W},
\end{aligned}
$$
with \(\lambda_{\text{var}} > 0\) controlling VaR aversion. VaR is typically estimated empirically from historical or simulated losses. However, VaR is not convex and fails to capture the **severity** of losses beyond the VaR threshold, motivating the use of Conditional VaR (CVaR). :contentReference[oaicite:2]{index=2}  

#### 3.1.4 Mean–CVaR portfolio optimization

The **Conditional Value‑at‑Risk (CVaR)** at level \(\alpha\) is
$$
\text{CVaR}_\alpha(L)=\mathbb{E}\big[\, L \mid L \ge \text{VaR}_\alpha(L) \,\big],
$$
i.e., the expected loss conditional on being in the worst \((1-\alpha)\) tail.

Rockafellar and Uryasev established an equivalent convex formulation:
$$
\text{CVaR}_\alpha(L)=\min_{\eta \in \mathbb{R}}
\left\{
\eta
+
\frac{1}{1-\alpha}\,
\mathbb{E}\big[(L - \eta)_+\big]
\right\},
$$
where \((x)_+ = \max(x, 0)\). For a finite set of loss scenarios \(L^{(s)}(w)\), \(s = 1,\dots,S\), this can be discretized as
$$
\begin{aligned}
\min_{w,\eta,u} \quad &
\eta + \frac{1}{(1-\alpha) S} \sum_{s=1}^S u_s, \\
\text{subject to} \quad &
u_s \ge 0,\quad u_s \ge L^{(s)}(w) - \eta,\quad s=1,\dots,S, \\
& \mathbf{1}^\top w = 1,\quad w \in \mathcal{W}.
\end{aligned}
$$

A static **mean–CVaR** portfolio problem balances expected return and tail risk:
$$
\begin{aligned}
\max_w \quad &
U_{\text{MCVaR}}(w)=w^\top \mu- \lambda_{\text{cvar}}\,
\text{CVaR}_\alpha\big(L(w)\big), \\
\text{subject to} \quad &
\mathbf{1}^\top w = 1,\quad w \in \mathcal{W}.
\end{aligned}
$$

CVaR is a coherent risk measure (convex, monotone, translation invariant, positively homogeneous) and is more sensitive to extreme losses than VaR. In our **dynamic** setting, this static mean–CVaR criterion is generalized to a per‑period reward for the PPO agent that subtracts a CVaR‑based penalty from realized monthly portfolio returns (Section 3.4.4). :contentReference[oaicite:3]{index=3}  

---

### 3.2 Stage 1: Predictive Modeling with XGBoost

Stage 1 estimates per‑ETF forward returns using gradient‑boosted regression trees (XGBoost). Each ETF has its own model, trained on a **unified feature set** of technical indicators and Fama–French–style factor and macro series. The unified set is selected once via a SHAP Gate (Section 3.3) and then reused across rolling windows. :contentReference[oaicite:4]{index=4}  

#### 3.2.1 Data, notation and labels

Let trading days be indexed by \(t = 1,\dots,T\) and ETFs by \(i \in \{1,\dots,N\}\). Let \(P_{i,t}\) denote the adjusted close price of ETF \(i\) on day \(t\). The simple **daily return** from day \(t\) to \(t+1\) is
$$
r_{i,t+1}= \frac{P_{i,t+1} - P_{i,t}}{P_{i,t}}.
$$

For a fixed horizon of \(H\) trading days (in experiments, \(H = 21\), approximately one month), we define the **forward simple return** label as the cumulative simple return over \(H\) days:
$$
y_{i,t}=\prod_{h=1}^{H} \bigl( 1 + r_{i,t+h} \bigr) - 1.
$$
This is the realized simple return from \(t\) (exclusive) to \(t+H\) (inclusive). We use raw forward returns \(y_{i,t}\) as labels; there is **no risk‑adjustment or residualization** in Stage 1. :contentReference[oaicite:5]{index=5}  

To avoid look‑ahead, we use **anchored rolling windows with purged splits**:

- For each anchor year, data are split chronologically into Train, Validation, and Test.  
- Because \(y_{i,t}\) depends on returns up to \(t+H\), the last \(H\) days of Train and Validation are **purged**, so labels never straddle into the next split.  
- An optional embargo of a few days may be added after each boundary.

This yields a sequence of walk‑forward experiments in which Stage 1 is retrained and evaluated on temporally disjoint data. :contentReference[oaicite:6]{index=6}  

#### 3.2.2 Feature design: technical indicators and Fama–French factors

For each ETF \(i\) and day \(t\), we construct a feature vector
$$
X_{i,t}=\bigl(
X^{\text{tech}}_{i,t},
X^{\text{ff}}_t,
X^{\text{macro}}_t
\bigr),
$$
where:

- \(X^{\text{tech}}_{i,t}\): **ETF‑specific technical indicators**, including:
  - Moving averages and price ratios at multiple horizons, e.g.  
    \(\text{SMA}_5\), \(\text{SMA}_{20}\), \(\text{SMA}_{60}\).
  - Momentum indicators: \(k\)‑day log returns  
    \[
    \text{MOM}^{(k)}_{i,t}    = \log P_{i,t} - \log P_{i,t-k},
    \]
    MACD‑type EMA spreads, and RSI‑style oscillators.
  - Volatility features, such as a 20‑day rolling standard deviation  
    \[
    \sigma^{(20)}_{i,t}    =    \sqrt{
      \frac{1}{20}
      \sum_{h=1}^{20}
      \bigl(
        r_{i,t+1-h}
        -
        \bar r^{(20)}_{i,t}
      \bigr)^2
    }.
    \]
  - Additional trend and mean‑reversion measures (e.g., moving‑average slopes).

- \(X^{\text{ff}}_t\): **Fama–French factor returns** and momentum, implemented as factor‑mimicking portfolios:
  $$
  X^{\text{ff}}_t  =  \bigl(
    \text{MKT}_t,\,
    \text{SMB}_t,\,
    \text{HML}_t,\,
    \text{RMW}_t,\,
    \text{CMA}_t,\,
    \text{UMD}_t,\,
    \dots
  \bigr).
  $$
  Each factor is included with lags \(\ell = 0,\dots,L_{\max}\):
  \(\,f_{k,t-\ell}\).

- \(X^{\text{macro}}_t\): additional macro variables (e.g., VIX, term spread, credit spreads), also lagged to avoid leak.

The **candidate** technical suffixes and factor families are fixed, and the SHAP Gate (Section 3.3) selects a small, stable subset that is then used in all rolling windows. :contentReference[oaicite:7]{index=7}  

#### 3.2.3 XGBoost model and objective

For each ETF \(i\), we fit a separate XGBoost regression model \(f_i(\cdot; \theta_i)\) mapping features \(X_{i,t}\) to the forward return forecast:
$$
\hat y_{i,t}=f_i(X_{i,t}; \theta_i)=
\sum_{k=1}^{K_i} f_{i,k}(X_{i,t}),
$$
where each \(f_{i,k}\) is a regression tree and \(\theta_i\) collects all tree parameters.

The training objective for ETF \(i\) on a given Train split is the regularized squared‑error loss
$$
\min_{\theta_i}
\left\{
\sum_{t \in \text{Train}}
\bigl( y_{i,t} - f_i(X_{i,t}; \theta_i) \bigr)^2
+
\sum_{k=1}^{K_i} \Omega\!\bigl(f_{i,k}\bigr)
\right\},
$$
where \(\Omega(\cdot)\) penalizes tree complexity (e.g., number of leaves and \(L_2\) leaf weights), following standard XGBoost practice. :contentReference[oaicite:8]{index=8}  

**Preprocessing.**

- Missing values in Train are imputed via feature‑wise medians using a `SimpleImputer`; the same imputer is applied to Validation and Test.
- Features are standardized via a Train‑fitted `StandardScaler` so all inputs share comparable scales.
- Hyperparameters (learning rate, max depth, subsampling, regularization) are tuned via a small random/grid search.

We select the configuration that maximizes the **time‑series Spearman rank correlation** (Information Coefficient, IC) between predictions and realized returns on Validation:
$$
\text{IC}=\operatorname{Corr}\Big(
  \operatorname{rank}(\hat y_{i,t}),
  \operatorname{rank}(y_{i,t})
\Big).
$$
Validation RMSE is used as a tie‑breaker. Each final model is retrained on Train+Validation and evaluated on Test, using an availability mask so that only samples with fully realized \(y_{i,t}\) (i.e., \(t + H\) within the data) are considered. :contentReference[oaicite:9]{index=9}  

#### 3.2.4 Outputs for downstream decision‑making

For each ETF \(i\) and test date \(t\), Stage 1 produces:

- **Predicted forward return:**
  \[
  \hat y_{i,t} = f_i(X_{i,t}).
  \]

- **Cross‑sectional percentile score:**
  $$
  \text{Score}_{i,t}  =
  \frac{
    \operatorname{rank}(\hat y_{i,t}) - 1
  }{
    N - 1
  },
  $$
  where \(\operatorname{rank}(\hat y_{i,t})\) is the rank of ETF \(i\)’s prediction among all \(N\) ETFs on date \(t\) (lowest \(= 1\), highest \(= N\)). Thus \(\text{Score}_{i,t} \in [0,1]\).

- **SHAP‑based feature attributions** (Section 3.3), both per‑feature and grouped by technical/factor families.

- **Realized forward return** \(y_{i,t}\), used for evaluation and for constructing RL rewards.

These outputs are stored in rolling files such as `stage1_predictions_with_shap_*.csv` and serve as the primary input stream to Stage 2. :contentReference[oaicite:10]{index=10}  

---

### 3.3 SHAP‑Based Explainability and Unified Feature Gate

Stage 1 models are tree ensembles, which allow efficient computation of **SHAP (Shapley Additive Explanations)** values via TreeSHAP. SHAP is used both to:

1. interpret per‑ETF forecasts, and  
2. construct a **Unified SHAP Gate** that selects a compact, cross‑ETF feature set and produces interpretable factor‑level attributions.   

#### 3.3.1 SHAP for tree ensembles

For ETF \(i\), model \(f_i\), and input \(X_{i,t}\), TreeSHAP computes the decomposition
$$
\hat y_{i,t}=f_i(X_{i,t})=\phi_{i,0}
+
\sum_{j=1}^{d} \phi_{i,j,t},
$$
where:

- \(\phi_{i,0}\) is the **base value** (expected model output over a background distribution),
- \(\phi_{i,j,t}\) is the **SHAP value** of feature \(j\) at input \(X_{i,t}\), representing the marginal contribution of feature \(j\) to the deviation of \(\hat y_{i,t}\) from \(\phi_{i,0}\).

SHAP values are grounded in cooperative game theory and satisfy local accuracy and consistency. For tree models, TreeSHAP yields exact Shapley values in polynomial time.   

We aggregate individual feature SHAP values into **feature‑group attributions**:

- For a technical suffix \(s\) (e.g., `SMA_5`), let \(G^{\text{tech}}(s)\) be all feature indices with this suffix for ETF \(i\). Group attribution:
  $$
  \Phi^{\text{tech}}_{i,s,t}  =  \sum_{j \in G^{\text{tech}}(s)} \phi_{i,j,t}.
  $$

- For a Fama–French factor family \(f\) (e.g., `MKT` with all lags), with indices \(G^{\text{ff}}(f)\), we define:
  $$
  \Phi^{\text{ff}}_{i,f,t}  =
  \sum_{j \in G^{\text{ff}}(f)} \phi_{i,j,t}.
  $$

These group‑level attributions quantify how much each technical or factor family contributed to the predicted forward return on date \(t\). :contentReference[oaicite:13]{index=13}  

#### 3.3.2 Unified SHAP Gate: global feature selection

To obtain a consistent, compact feature set across ETFs and windows, we run a **Unified SHAP Gate** on a fixed **base window** \(\mathcal{T}_{\text{base}}\) prior to any validation/test periods.

1. **Base window and candidate pool.**  
   On \(\mathcal{T}_{\text{base}}\), for each ETF \(i\) we build an initial feature pool consisting of:
   - all ETF‑specific technical features for that ETF (all available suffixes \(s\)), and
   - all macro/factor features (Fama–French families: MKT, SMB, HML, RMW, CMA, UMD, etc.) with lags \(\ell = 0,\dots,L_{\max}\).

2. **Gate models and mean absolute SHAP importance.**  
   For each ETF \(i\), we fit a **small XGBoost regressor** \(g_i\) on the base window, using squared‑error loss and shallow trees for interpretability. TreeSHAP yields SHAP values \(\phi^{\text{gate}}_{i,j,t}\). Define the **mean absolute SHAP importance** of feature \(j\) for ETF \(i\) as
   $$
   I_i(j)   =   \mathbb{E}_{t \in \mathcal{T}_{\text{base}}}
   \bigl[\,|\phi^{\text{gate}}_{i,j,t}|\,\bigr].
   $$

3. **Aggregation over technical suffixes and factor families.**  

   - For each technical suffix \(s\), ETF \(i\) has a feature \(j(i,s)\) with that suffix. The cross‑ETF importance is
     $$
     I(s)     =
     \frac{1}{|U|}
     \sum_{i \in U} I_i\bigl(j(i,s)\bigr),
     $$
     where \(U\) is the ETF universe.

   - For each factor family \(f\), with feature set \(F(f)\) (all lags), we define:
     $$
     \bar I_i(f)     =     \frac{1}{|F(f)|}     \sum_{j \in F(f)} I_i(j), \qquad     I(f)     =
     \frac{1}{|U|}
     \sum_{i \in U} \bar I_i(f).
     $$

4. **Selection under budget constraints.**  

   Technical suffix groups \(\{s\}\) are ranked by \(I(s)\), and factor families \(\{f\}\) by \(I(f)\). Given budget parameters — at least \(\text{MIN\_SUFFIXES}\) technical groups, at most \(\text{MAX\_GLOBAL\_FAMILIES}\) factor families, and a total of \(K\) groups — we select:

   - the top \(\text{MIN\_SUFFIXES}\) technical suffixes by \(I(s)\),
   - up to \(\text{MAX\_GLOBAL\_FAMILIES}\) factor families by \(I(f)\),
   - additional groups in decreasing order of score until \(K\) groups are selected.

   For each selected factor family \(f\), we retain only the single lag with highest cross‑ETF importance, reducing redundancy. For each chosen technical suffix \(s\), we keep that indicator for all ETFs. The resulting **unified feature set** is frozen and used in all rolling windows. :contentReference[oaicite:14]{index=14}  

This gate yields a stable, data‑driven feature selection that reflects both technical and factor relevance while controlling model complexity.

#### 3.3.3 SHAP‑based diagnostics and derived measures

Beyond feature selection, we derive additional SHAP‑based diagnostics:

- **Global group importance** for group \(g\) across ETFs and time:
  $$
  \bar I(g)  =  \mathbb{E}_{i,t}
  \bigl[ |\Phi_{i,g,t}| \bigr],
  $$
  which ranks technical and factor families by their typical contribution magnitude.

- **Relative (demeaned) SHAP values** highlighting cross‑sectional deviations:
  $$
  \mathrm{REL\_SHAP}_{i,g,t}  =  \Phi_{i,g,t}  -  \frac{1}{N}
  \sum_{j=1}^{N}
  \Phi_{j,g,t},
  $$
  indicating whether group \(g\) was more or less influential for ETF \(i\) than for its peers on date \(t\).

- **Consistency check:** for each \((i,t)\), SHAP attributions satisfy
  $$
  \phi_{i,0}  +  \sum_{j} \phi_{i,j,t}  =  \hat y_{i,t},
  $$
  ensuring exact reconstruction of the model prediction.

These diagnostics create an interpretable bridge between Fama–French factors, technical indicators, and the Stage 1 forecasts. Selected SHAP metrics can be passed to Stage 2 as part of the observation state, or used offline to analyze the learned policy. :contentReference[oaicite:15]{index=15}  

---

### 3.4 Stage 2: PPO‑Based Deep Reinforcement Learning Agent

Stage 2 maps Stage 1 signals and lagged market features into dynamic portfolio allocations using a Proximal Policy Optimization (PPO) agent. The environment is modeled as a Markov Decision Process (MDP) with continuous actions and a reward that generalizes mean–CVaR portfolio optimization to a sequential setting.   

#### 3.4.1 Environment and MDP definition

Let \(\mathcal{U} = \{1,\dots,N\}\) be the ETF universe. For each trading day \(t\), we observe daily return vector \(r_{t+1} \in \mathbb{R}^N\) with elements \(r_{i,t+1}\). Portfolio weights \(w_t \in \mathbb{R}^N\) satisfy \(\mathbf{1}^\top w_t = 1\) and lie in a feasible set \(\mathcal{W}\) (long‑only with per‑ETF bounds in our experiments).

The MDP components are:

- **State \(S_t\):** a stacked history of multi‑ETF features and Stage 1 outputs up to day \(t\).
- **Action \(A_t\):** a continuous vector in \([-1,1]^N\) specifying tilts relative to a baseline portfolio at monthly rebalancing times.
- **Transition:** between rebalances, portfolio weights evolve with asset returns in a self‑financing way (Section 3.4.3.3).
- **Reward \(R_t\):** a monthly performance measure combining realized return, CVaR‑based tail risk penalty, and regularization terms (turnover, concentration, transaction costs).

The environment uses a **monthly wrapper**: the agent acts every \(\Delta = 21\) trading days (approx. one month). At time \(t\), the agent chooses an action \(A_t\); the environment simulates \(\Delta\) daily steps, updating portfolio weights via daily returns, and returns a single aggregate reward \(R_t\) summarizing performance over that month.

#### 3.4.2 Observation space

At each monthly decision time \(t\), the observation \(S_t\) aggregates the last \(L\) days of features:
$$
S_t=\bigl[
  Z_{t-L+1},
  Z_{t-L+2},
  \dots,
  Z_t
\bigr],
$$
where \(Z_\tau\) is the concatenation (across ETFs) of:

- Stage 1 outputs when available (e.g., \(\hat y_{i,\tau}\), \(\text{Score}_{i,\tau}\)),
- selected technical indicators such as momentum and volatility (aligned with Stage 1’s unified feature set),
- Fama–French factors and macro variables (appropriately lagged),
- optional SHAP‑based group metrics (e.g., \(\Phi^{\text{ff}}_{i,f,\tau}\), \(\mathrm{REL\_SHAP}_{i,g,\tau}\)),
- portfolio‑level summaries (e.g., current weights, recent portfolio returns).

All features in \(S_t\) are constructed using information up to day \(t\); no future returns enter the state, and Stage 2 uses only lagged/unleaked features. :contentReference[oaicite:17]{index=17}  

#### 3.4.3 Action space and mapping to portfolio weights

The PPO policy outputs
$$
A_t=(a_{1,t}, \dots, a_{N,t})^\top
\in [-1,1]^N,
$$
interpreted as **baseline‑relative tilts**. Let \(w_t^{(b)}\) denote the **baseline portfolio** at time \(t\) (Section 3.4.3.1). The raw post‑action weights are
$$
w^{\text{raw}}_{i,t}=w^{(b)}_{i,t}
+
a_{i,t} \, w^{(b)}_{i,t},
$$
so that \(a_{i,t} = 0\) leaves ETF \(i\) at its baseline weight, \(a_{i,t} > 0\) overweights it, and \(a_{i,t} < 0\) underweights it. The mapping from \(A_t\) to feasible weights \(w_t\) has three steps.

##### 3.4.3.1 Baseline portfolio construction

The baseline weights \(w_t^{(b)}\) are constructed from **score vectors** \(s_{i,t}\) that summarize signal strength, typically combining Stage 1 cross‑sectional scores and momentum/volatility indicators:

1. **Scores.** Assign a scalar score \(s_{i,t}\) to each ETF, for example:
   - \(s_{i,t} = \text{Score}_{i,t}\) (Stage 1 percentile score), or
   - a composite of technical momentum signals.

2. **Risk adjustment.** Scale scores by inverse volatility, e.g.
   $$
   \tilde s_{i,t}   =   \frac{s_{i,t}}{\sigma_{i,t}},
   $$
   where \(\sigma_{i,t}\) is a recent volatility estimate (e.g., 20‑day \(\sigma^{(20)}_{i,t}\)), to down‑weight high‑volatility ETFs.

3. **Adaptive softmax weights.** Let \(\bar s_t\) and \(\operatorname{sd}(s_t)\) be the cross‑sectional mean and standard deviation of \(\tilde s_{i,t}\). We define a temperature \(\tau_t\) proportional to \(\operatorname{sd}(s_t)\), and set:
   $$
   w^{(b)}_{i,t}   =   \frac{
     \exp\big((\tilde s_{i,t} - \bar s_t) / \tau_t\big)
   }{
     \sum_{j=1}^{N}
     \exp\big((\tilde s_{j,t} - \bar s_t) / \tau_t\big)
   }.
   $$
   This yields a **softmax‑tilted**, risk‑adjusted portfolio: large positive scores get higher weights, but dispersion is controlled via \(\tau_t\).

4. **Floor/cap and smoothing.** Enforce per‑ETF floors and caps (e.g., \(w_{\min} = 0.02\), \(w_{\max} = 0.22\)), renormalize, and apply an exponential moving average in time to smooth \(w^{(b)}_t\).

The baseline thus resembles a **factor/momentum tilt strategy** and serves as both a reference portfolio and a prior around which the RL agent makes relative bets. :contentReference[oaicite:18]{index=18}  

##### 3.4.3.2 Budget and constraint projection

The raw weights \(w^{\text{raw}}_t\) may violate budget or bound constraints. Let \(B_{\text{long}}\) and \(B_{\text{short}}\) be the total long and short budgets (e.g., \(B_{\text{long}} = 1\), \(B_{\text{short}} = 0\) for long‑only). Define
$$
w^+_{i,t} = \max(w^{\text{raw}}_{i,t}, 0),
\qquad
w^-_{i,t} = \min(w^{\text{raw}}_{i,t}, 0).
$$

We scale these to enforce budgets:

- If \(\sum_i w^+_{i,t} > B_{\text{long}}\), set
  $$
  w^+_{i,t}
  \leftarrow
  \frac{B_{\text{long}}}{\sum_j w^+_{j,t}}\, w^+_{i,t}.
  $$
- If \(-\sum_i w^-_{i,t} > B_{\text{short}}\), set
  $$
  w^-_{i,t}
  \leftarrow
  \frac{B_{\text{short}}}{-\sum_j w^-_{j,t}}\, w^-_{i,t}.
  $$

The combined weights are
$$
w_{i,t}=
w^+_{i,t} + w^-_{i,t},
$$
which are then projected onto individual bounds (e.g., \([0,1]\) for long‑only) and renormalized if necessary. The resulting vector \(w_t\) is the **feasible** post‑trade portfolio. :contentReference[oaicite:19]{index=19}  

##### 3.4.3.3 Self‑financing portfolio dynamics

Between rebalancing times, the portfolio is held without trading. If \(w_t\) are the weights at day \(t\) and \(r_{t+1}\) is the vector of ETF returns, the self‑financing weight update is
$$
w_{t+1}=
\frac{
  w_t \odot (1 + r_{t+1})
}{
  1 + w_t^\top r_{t+1}
},
$$
where \(\odot\) denotes elementwise multiplication. This preserves the fully invested constraint \(\mathbf{1}^\top w_{t+1} = 1\) and ensures no external capital is injected or withdrawn. :contentReference[oaicite:20]{index=20}  

#### 3.4.4 Reward: mean–CVaR objective with penalties

Let \(R^{\text{daily}}_\tau\) be the **daily portfolio return** on day \(\tau\). For a monthly decision interval \([t+1, t+\Delta]\) of \(\Delta = 21\) days, the compounded monthly return is
$$
R^{\text{month}}_t=
\prod_{\tau = t+1}^{t+\Delta}
 (1 + R^{\text{daily}}_\tau)
 - 1.
$$
Define the monthly loss
$$
L_t=
- R^{\text{month}}_t.
$$

From the history \(\{L_u\}_{u < t}\), we estimate empirical CVaR at level \(\alpha\) (e.g., \(\alpha = 0.95\)):

- Let \(M\) be the number of historical monthly losses and \(\{ L_t^{(1)} \ge \dots \ge L_t^{(M)} \}\) the losses sorted from worst to best.
- Let \(K = \lfloor (1-\alpha) M \rfloor\). Then
  $$
  \widehat{\text{CVaR}}_\alpha(L_t)  =  \frac{1}{K}
  \sum_{k=1}^{K} L_t^{(k)}.
  $$

We define a **mean–CVaR base reward**
$$
\tilde R_t=
R^{\text{month}}_t-
\lambda_{\text{cvar}}\,
\widehat{\text{CVaR}}_\alpha(L_t),
$$
with \(\lambda_{\text{cvar}} > 0\) controlling tail‑risk aversion.

To better capture practical considerations, we add:

- **Turnover penalty.** Define one‑way turnover at rebalancing as
  $$
  T_t  =  \frac{1}{2}
  \sum_{i=1}^N
  \bigl| w_{i,t} - w_{i,t^-} \bigr|,
  $$
  where \(w_{t^-}\) are pre‑trade weights. This is penalized via \(\lambda_T T_t\).

- **Concentration penalty.** Use the Herfindahl–Hirschman Index (HHI):
  $$
  H_t  =
  \sum_{i=1}^N w_{i,t}^2,
  $$
  penalized via \(\lambda_H H_t\) to encourage diversification.

- **Transaction costs.** Proportional to turnover: \(c_{\text{tc}} T_t\).

- **(Optional) deviation from baseline.** A penalty \(\lambda_B \| w_t - w^{(b)}_t \|_1\); in the main configuration we set \(\lambda_B = 0\).

The final monthly reward is
$$
R_t=
R^{\text{month}}_t-\lambda_{\text{cvar}}\,\widehat{\text{CVaR}}_\alpha(L_t)-\lambda_T T_t-\lambda_H H_t-c_{\text{tc}} T_t-
\lambda_B \| w_t - w^{(b)}_t \|_1.
$$

This reward function is a **dynamic analogue** of static mean–CVaR optimization, augmented with implementation and diversification penalties. :contentReference[oaicite:21]{index=21}  

#### 3.4.5 PPO policy optimization

Let \(\pi_\theta(a \mid s)\) be the stochastic policy with parameters \(\theta\), and let \(V_\phi(s)\) be the value function with parameters \(\phi\). We train both with **Proximal Policy Optimization (PPO)**, an on‑policy actor–critic algorithm using a clipped surrogate loss. :contentReference[oaicite:22]{index=22}  

Given trajectories \(\{(S_t, A_t, R_t)\}\), we compute advantage estimates \(\hat A_t\) using generalized advantage estimation (GAE). The importance sampling ratio is
$$
r_t(\theta)=
\frac{
  \pi_\theta(A_t \mid S_t)
}{
  \pi_{\theta_{\text{old}}}(A_t \mid S_t)
}.
$$

The clipped surrogate objective is
$$
L_{\text{clip}}(\theta)=\mathbb{E}_t
\left[
  \min\!\Big(
    r_t(\theta)\, \hat A_t,\,
    \operatorname{clip}\big(r_t(\theta), 1-\varepsilon, 1+\varepsilon\big)\, \hat A_t
  \Big)
\right],
$$
with clip parameter \(\varepsilon \approx 0.2\). The full PPO loss combines this with value loss and entropy regularization:
$$
L_{\text{PPO}}(\theta, \phi)=- L_{\text{clip}}(\theta)+c_V \,\mathbb{E}_t\big[
  ( V_\phi(S_t) - R_t^{\text{target}} )^2
\big]-
c_H \,
\mathbb{E}_t
\big[
  \mathcal{H}(\pi_\theta(\cdot \mid S_t))
\big],
$$
where \(R_t^{\text{target}}\) is a return target (e.g., \(\lambda\)‑discounted return), \(\mathcal{H}(\cdot)\) is the policy entropy, and \(c_V, c_H\) are weights.  

Both policy and value networks are implemented as multilayer perceptrons with two hidden layers (e.g., 64 ReLU units each). We use Adam for optimization, and apply reward normalization (but not observation normalization) via a `VecNormalize`‑style wrapper to stabilize training while preserving economically meaningful feature scales. :contentReference[oaicite:23]{index=23}  

---

### 3.5 Combined Hierarchical Signal‑to‑Policy Framework

Bringing the components together, we obtain a **SHAP‑informed hierarchical signal‑to‑policy framework** that integrates Fama–French theory, supervised ML prediction, and risk‑aware DRL into a single pipeline.   

#### 3.5.1 Conceptual structure

The overall architecture has four interacting layers:

1. **Factor‑based theoretical layer (Section 3.1).**  
   Fama–French multi‑factor models provide an economic lens on ETF returns, and mean–variance / mean–VaR / mean–CVaR formulations formalize risk–return trade‑offs.

2. **Stage 1: predictive layer (Section 3.2).**  
   Per‑ETF XGBoost models \(f_i\) predict forward returns \(y_{i,t}\) from a unified set of technical and factor features \(X_{i,t}\). Anchored rolling windows with purged splits enforce realistic out‑of‑sample evaluation.

3. **Explainable layer and SHAP Gate (Section 3.3).**  
   TreeSHAP decomposes predictions into per‑feature and per‑group contributions; these are aggregated into technical and factor families. The Unified SHAP Gate produces a compact, stable feature set and global importance rankings.

4. **Stage 2: decision layer (Section 3.4).**  
   A PPO agent observes a history of Stage 1 signals and market features, outputs tilt actions \(A_t\), and is trained under a dynamic mean–CVaR objective with penalties for turnover, concentration, and transaction costs.

This hierarchy mirrors institutional workflows: a **research layer** that generates interpretable signals, followed by a **portfolio construction layer** that manages risk and implementation details.

#### 3.5.2 Mathematical integration of stages

The composite mapping from raw data to portfolio weights can be summarized as follows.

1. **Signal generation (Stage 1).**  
   For ETF \(i\) and decision date \(t\), Stage 1 computes:
   $$
   \hat y_{i,t} = f_i(X_{i,t}), \qquad
   \text{Score}_{i,t}, \qquad
   \{ \Phi_{i,g,t} \}_{g \in \mathcal{G}},
   $$
   where \(\mathcal{G}\) indexes technical and factor families. These are functions of past prices and Fama–French factors only.

2. **State construction (Stage 2).**  
   The RL state at time \(t\) is
   $$
   S_t   =   \bigl[
     S_{t-L+1}^{\text{raw}},\,
     \dots,\,
     S_t^{\text{raw}}
   \bigr],
   $$
   where each \(S_\tau^{\text{raw}}\) concatenates per‑ETF Stage 1 outputs, selected technical indicators, factor/macro features, and possibly SHAP metrics.

3. **Policy mapping and weight construction.**  
   The PPO policy \(\pi_\theta\) maps \(S_t\) to a distribution over tilt actions \(A_t\). A baseline map \(b(\cdot)\) converts Stage 1 scores (and other signals) to baseline weights \(w_t^{(b)} = b(\{\text{Score}_{i,t}\})\). Final weights are:
   $$
   w_t   =
   \Pi_{\mathcal{W}}
   \bigl(
     w_t^{(b)}
     + a_t \odot w_t^{(b)}
   \bigr),
   $$
   where \(\Pi_{\mathcal{W}}\) is the projection onto budget and bound constraints, and \(\odot\) is elementwise multiplication.

4. **Dynamic mean–CVaR objective.**  
   The induced sequence \(\{R^{\text{month}}_t\}\) of monthly returns and associated losses \(\{L_t\}\) defines a stochastic process under \(\pi_\theta\). The PPO training objective approximates:
   $$
   \max_{\pi}
   \;
   \mathbb{E}_\pi
   \left[
     \sum_{t=0}^{T-1}
     \gamma^t
     \Big(
       R^{\text{month}}_t
       -
       \lambda_{\text{cvar}}\,
       \widehat{\text{CVaR}}_\alpha(L_t)
       -
       \lambda_T T_t
       -
       \lambda_H H_t
       -
       c_{\text{tc}} T_t
     \Big)
   \right],
   $$
   with discount factor \(\gamma \lesssim 1\). This generalizes the static mean–CVaR criterion (Section 3.1.4) to a sequential setting, where the policy jointly shapes returns, tail losses, and trading behavior over time.

#### 3.5.3 Advantages of the hierarchical design

The proposed architecture offers several methodological advantages:

- **Separation of prediction and allocation.**  
  Stage 1 focuses on forecasting returns with interpretable features; Stage 2 focuses on risk‑aware allocation. This modularity simplifies diagnostics and allows each layer to be improved independently.

- **Factor‑aware interpretability.**  
  SHAP values aggregated by Fama–French factor families and technical groups provide intuitive decompositions of both forecasts and realized portfolio performance, aiding economic interpretation.

- **Dynamic mean–CVaR optimization.**  
  The PPO agent optimizes a reward that directly encodes mean–CVaR preferences with turnover and concentration penalties, extending classical static optimization to a data‑driven, dynamic environment.

- **Data efficiency and stability.**  
  Stage 1 distills noisy raw data into structured signals (predicted returns, scores, SHAP attributions), reducing the burden on the RL agent, which can focus on capital allocation under risk and cost constraints rather than relearning predictive structure.

- **Alignment with practice.**  
  The hierarchy aligns with how institutional investors operate: research teams generate factor‑aware views and signals, while portfolio managers or optimizers implement them under explicit risk, cost, and constraint frameworks.

In summary, the SHAP‑informed signal‑to‑policy framework ties together Fama–French factor theory, modern machine learning, and deep reinforcement learning into a unified methodology for dynamic, risk‑aware portfolio optimization.   
