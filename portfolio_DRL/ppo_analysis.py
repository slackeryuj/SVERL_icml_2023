
# ppo_analysis.py — Integrated analytics and charts
# =================================================
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from scipy import stats
import re

mpl.rcParams['figure.figsize'] = (7.5, 5.0)
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['savefig.dpi'] = 500

TRADING_DAYS = 252

LABEL_MAP = {
    "SHAP+MOM+Stage1": "SHAP Informed PPO Model",
    "MOM+Stage1": "PPO Model",
    "MeanCVaR": "Mean-CVaR with Penalty",
    "Equal Weight": "Equal Weight",
}

DISPLAY_MAP = {
    "SHAP":        "SHAP Informed PPO Model",
    "PPO":         "PPO Model",
    "MeanCVaR":    "Mean-CVaR with Penalty",
    "EqualWeight": "Equal Weight",
}

def wrap_label(s: str) -> str:
    if s == "Mean-CVaR with Penalty":
        return "Mean-CVaR\nwith Penalty"
    if s == "SHAP Informed PPO Model":
        return "SHAP Informed\nPPO Model"
    return s

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def best_match(available_sheets, candidates):
    nm = {norm(s): s for s in available_sheets}
    for cand in candidates:
        k = norm(cand)
        if k in nm:
            return nm[k]
        for kk, vv in nm.items():
            if k in kk:
                return vv
    return available_sheets[0]

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def strip_summary_rows(df: pd.DataFrame, min_daily_rows: int = 100) -> pd.DataFrame:
    df2 = df.copy()
    if df2.shape[0] > 10:
        df2 = df2.iloc[:-3, :]
    df2 = df2.dropna(how="all")
    if df2.shape[0] < min_daily_rows:
        df2 = df.dropna(how="all")
    return df2

def load_returns_df(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = coerce_numeric_df(xls.parse(sheet_name))
    df = strip_summary_rows(df)
    drop_cols = []
    for c in df.columns:
        ser = df[c].dropna()
        if ser.empty or ser.nunique() <= max(1, int(0.05 * len(ser))):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.shape[1] == 0:
        return df
    med_len = int(np.median([df[c].notna().sum() for c in df.columns]))
    keep = [c for c in df.columns if df[c].notna().sum() >= 0.5 * med_len] if med_len>0 else list(df.columns)
    df = df[keep] if keep else df
    df = df.dropna(how="any")
    df.columns = [f"Iter_{i+1}" for i in range(df.shape[1])]
    return df


def compute_cagr(daily: pd.Series) -> float:
    n = daily.shape[0]
    tot = float((1.0 + daily).prod() - 1.0)
    return (1.0 + tot) ** (TRADING_DAYS / n) - 1.0

def compute_ann_vol(daily: pd.Series) -> float:
    return float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))

def compute_return_vol_ratio(daily: pd.Series) -> float:
    vol = compute_ann_vol(daily)
    return float(compute_cagr(daily) / vol) if vol else np.nan

def compute_sharpe(daily: pd.Series, rf: float = 0.0) -> float:
    mu = float(daily.mean()) * TRADING_DAYS
    vol = compute_ann_vol(daily)
    return float((mu - rf) / vol) if vol else np.nan

def compute_sortino(daily: pd.Series) -> float:
    downside = daily[daily < 0.0]
    if downside.shape[0] == 0:
        return np.inf
    dd = float(downside.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    mu = float(daily.mean()) * TRADING_DAYS
    return float(mu / dd) if dd else np.nan

def compute_max_drawdown(daily: pd.Series) -> float:
    cr = (1.0 + daily).cumprod()
    peak = cr.cummax()
    dd = cr / peak - 1.0
    return float(dd.min())

def compute_calmar(daily: pd.Series) -> float:
    cagr = compute_cagr(daily)
    mdd = compute_max_drawdown(daily)
    denom = abs(mdd) if mdd != 0 else np.nan
    return float(cagr / denom) if denom else np.nan

def block_monthly_returns(daily: pd.Series, block_len: int = 21) -> pd.Series:
    g = pd.Series(np.arange(len(daily)) // block_len, index=daily.index)
    return (1.0 + daily).groupby(g).prod() - 1.0

def compute_metrics_for_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col].astype(float); monthly = block_monthly_returns(s, 21)
        rows.append({
            "iteration": col, "n_days": s.shape[0],
            "CAGR": compute_cagr(s), "AnnVol": compute_ann_vol(s),
            "Return/Vol": compute_return_vol_ratio(s), "Sharpe": compute_sharpe(s),
            "Sortino": compute_sortino(s), "MaxDD": compute_max_drawdown(s),
            "Calmar": compute_calmar(s),
            "Monthly_CVaR_5%": float(monthly[monthly <= monthly.quantile(0.05)].mean() if len(monthly)>0 else np.nan),
        })
    return pd.DataFrame(rows)


def finalize_fig(title: str, caption: str = None, legend_outside: bool = False):
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    plt.title(title)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        if legend_outside:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
        else:
            ax.legend(loc='best', frameon=True)
    if caption:
        plt.figtext(0.01, -0.08, caption, ha='left', va='top', fontsize=10)
    plt.tight_layout()

def percentify_axis(axis='y', decimals=1):
    ax = plt.gca()
    fmt = PercentFormatter(xmax=1.0, decimals=decimals)
    if axis == 'y':
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

def savefig_both(path_png: Path):
    path_pdf = Path(path_png).with_suffix(".pdf")
    plt.savefig(Path(path_png).as_posix(), dpi=500, bbox_inches="tight")
    plt.savefig(path_pdf.as_posix(), bbox_inches="tight")
    plt.close()

def remap_keys(d: dict) -> dict:
    return {LABEL_MAP.get(k, k): v for k,v in d.items()}


def cum_returns_plot(ensemble_by_method: dict, universe_label: str, out_file: Path):
    plt.figure()
    ens_map = remap_keys(ensemble_by_method)
    for m, ens in ens_map.items():
        cr = (1.0 + ens).cumprod()
        plt.plot(cr.values, label=m, linewidth=1.8)
        plt.text(len(cr.values)-1, cr.values[-1], f"  {m}: {cr.values[-1]:.2f}×", va='center')
    plt.xlabel("Trading Day"); plt.ylabel("Cumulative Growth of $1")
    finalize_fig(
        title=f"{universe_label}: Cumulative Returns (Median Ensemble)",
        caption="Each series is the median across 30 PPO runs, compounded from daily returns.",
        legend_outside=True
    )
    savefig_both(out_file)

def rolling_sharpe_plot(ensemble_by_method: dict, universe_label: str, window: int, out_file: Path):
    plt.figure()
    ens_map = remap_keys(ensemble_by_method)
    for m, ens in ens_map.items():
        s = ens.astype(float)
        mu = s.rolling(window).mean() * TRADING_DAYS
        vol = s.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        rs = mu / vol
        plt.plot(rs.values, label=m, linewidth=1.5, alpha=0.95)
    plt.xlabel("Trading Day"); plt.ylabel("Rolling Sharpe (annualized)")
    finalize_fig(
        title=f"{universe_label}: Rolling Sharpe ({window} trading days) — Median Ensembles",
        caption="Sharpe computed from rolling mean and std of daily returns (annualized).",
        legend_outside=True
    )
    savefig_both(out_file)


def boxplot_metric(metrics_by_method: dict, metric: str, universe_label: str, ylabel: str, out_file: Path, percent_axis=False):
    plt.figure()
    met_map = remap_keys(metrics_by_method)
    data = [met_map[m][metric].values for m in met_map.keys()]
    labels = [wrap_label(m) for m in met_map.keys()]
    try:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=0); plt.xlabel("Model"); plt.ylabel(ylabel)
    if percent_axis: percentify_axis('y', decimals=1)
    finalize_fig(
        title=f"{universe_label}: {metric} — Distribution Across 30 Iterations",
        caption="Boxes: IQR; center line: median; diamond = mean; whiskers: non-outlier range.",
        legend_outside=False
    )
    savefig_both(out_file)

def bar_means_ci(metrics_by_method: dict, metric: str, universe_label: str, ylabel: str, out_file: Path, percent_axis=False):
    plt.figure()
    met_map = remap_keys(metrics_by_method)
    methods = list(met_map.keys())
    means, cis = [], []
    for m in methods:
        x = met_map[m][metric].dropna().values
        if len(x) == 0:
            means.append(np.nan); cis.append(0.0); continue
        mean = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x)>1 else 0.0
        ci = 1.96 * se
        means.append(mean); cis.append(ci)
    xloc = np.arange(len(methods))
    plt.bar(xloc, means, yerr=cis, capsize=4)
    xtick_labels = [wrap_label(m) for m in methods]
    plt.xticks(xloc, xtick_labels, rotation=0)
    plt.xlabel("Model"); plt.ylabel(ylabel)
    if percent_axis:
        percentify_axis('y', decimals=1)
        for i, mean in enumerate(means):
            if not (mean is None or np.isnan(mean)):
                plt.text(i, mean + (cis[i] if i < len(cis) else 0) + 0.005, f"{mean*100:.1f}%", ha='center', va='bottom')
    else:
        for i, mean in enumerate(means):
            if not (mean is None or np.isnan(mean)):
                plt.text(i, mean + (cis[i] if i < len(cis) else 0) + 0.01, f"{mean:.2f}", ha='center', va='bottom')
    finalize_fig(
        title=f"{universe_label}: Mean ± 95% CI — {metric}",
        caption="Bars show mean across 30 iterations; error bars are 95% normal-approximate CIs.",
        legend_outside=False
    )
    savefig_both(out_file)


def scatter_tradeoff(metrics_by_method: dict, x_metric: str, y_metric: str,
                     universe_label: str, xlabel: str, ylabel: str, out_file: Path,
                     x_percent=False, y_percent=False, annotate_means=True):
    plt.figure()
    met_map = remap_keys(metrics_by_method)
    for m in met_map.keys():
        dfm = met_map[m]
        plt.scatter(dfm[x_metric].values, dfm[y_metric].values, label=m, alpha=0.65)
        if annotate_means:
            mx, my = float(dfm[x_metric].mean()), float(dfm[y_metric].mean())
            plt.scatter([mx], [my], marker='X', s=120)
            plt.text(mx, my, f"  {wrap_label(m)} (mean)", va='center')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if x_percent: percentify_axis('x', decimals=1)
    if y_percent: percentify_axis('y', decimals=1)
    finalize_fig(
        title=f"{universe_label}: {ylabel} vs {xlabel}",
        caption="Each point is one PPO run; 'X' marks the model mean across 30 runs.",
        legend_outside=True
    )
    savefig_both(out_file)

def independent_tests(x: np.ndarray, y: np.ndarray):
    t_res = stats.ttest_ind(x, y, equal_var=False)
    mwu = stats.mannwhitneyu(x, y, alternative="greater")
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1)*vx + (ny - 1)*vy) / (nx + ny - 2) if (nx+ny-2)>0 else np.nan
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pooled) if (pooled and pooled>0) else np.nan
    U = mwu.statistic; p_sup = float(U / (len(x) * len(y)))
    return dict(t_stat=float(t_res.statistic), t_pvalue=float(t_res.pvalue),
                mwu_U=float(U), mwu_pvalue=float(mwu.pvalue),
                cohen_d=float(d), P_sup=float(p_sup))

def build_pairwise_df(metrics_by_method: dict) -> pd.DataFrame:
    met_map = remap_keys(metrics_by_method)
    target = "SHAP Informed PPO Model"
    comps = [m for m in met_map.keys() if m != target]
    rows = []
    for comp in comps:
        for metric in ["CAGR","Return/Vol","Sharpe","Sortino","MaxDD","Calmar","Monthly_CVaR_5%"]:
            x = met_map[target][metric].dropna().values
            y = met_map[comp][metric].dropna().values
            if len(y) <= 1:
                if len(y) == 1:
                    wins = int(np.sum(x > y[0])); n = len(x)
                    bt = stats.binomtest(wins, n, p=0.5, alternative="greater")
                    rows.append(dict(Compare=f"{target} vs {comp}", Metric=metric, binom_k=wins, binom_n=n, binom_pvalue=float(bt.pvalue)))
                else:
                    rows.append(dict(Compare=f"{target} vs {comp}", Metric=metric, binom_k=np.nan, binom_n=np.nan, binom_pvalue=np.nan))
            else:
                res = independent_tests(x, y)
                rows.append(dict(Compare=f"{target} vs {comp}", Metric=metric, mwu_pvalue=res["mwu_pvalue"], P_sup=res["P_sup"], cohen_d=res["cohen_d"]))
    return pd.DataFrame(rows)


def moving_block_bootstrap_indices(n: int, block_len: int, B: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    starts = np.arange(0, max(1, n - block_len + 1))
    n_blocks = int(np.ceil(n / block_len))
    for _ in range(B):
        idx = []
        for _ in range(n_blocks):
            s = int(rng.choice(starts, 1))
            idx.extend(range(s, min(s + block_len, n)))
        yield np.array(idx[:n], dtype=int)

def bootstrap_ensemble_diffs(a: pd.Series, b: pd.Series, block_len: int = 21, B: int = 2000, seed: int = 42):
    a = a.reset_index(drop=True).astype(float)
    b = b.reset_index(drop=True).astype(float)
    n = min(len(a), len(b))
    a, b = a.iloc[:n], b.iloc[:n]
    diffs_mean, diffs_sharpe, diffs_rvr = [], [], []
    for idx in moving_block_bootstrap_indices(n, block_len, B, seed):
        ra = a.iloc[idx]; rb = b.iloc[idx]
        diffs_mean.append(float(ra.mean() - rb.mean()))
        mu_a, mu_b = float(ra.mean())*TRADING_DAYS, float(rb.mean())*TRADING_DAYS
        vol_a, vol_b = float(ra.std(ddof=1))*np.sqrt(TRADING_DAYS), float(rb.std(ddof=1))*np.sqrt(TRADING_DAYS)
        sh_a = (mu_a/vol_a) if vol_a else np.nan; sh_b = (mu_b/vol_b) if vol_b else np.nan
        diffs_sharpe.append(float(sh_a - sh_b))
        def cagr(x):
            tot = float((1.0 + x).prod() - 1.0)
            return (1.0 + tot)**(TRADING_DAYS/len(x)) - 1.0
        rvr_a = cagr(ra) / (float(ra.std(ddof=1))*np.sqrt(TRADING_DAYS)) if float(ra.std(ddof=1)) else np.nan
        rvr_b = cagr(rb) / (float(rb.std(ddof=1))*np.sqrt(TRADING_DAYS)) if float(rb.std(ddof=1)) else np.nan
        diffs_rvr.append(float(rvr_a - rvr_b))
    diffs_mean = np.array(diffs_mean); diffs_sharpe = np.array(diffs_sharpe); diffs_rvr = np.array(diffs_rvr)
    p = lambda arr: float(2.0 * min((arr <= 0).mean(), (arr >= 0).mean()))
    return dict(
        mean_diff=float(np.nanmean(diffs_mean)),
        mean_diff_p=p(diffs_mean),
        mean_diff_ci_low=float(np.nanquantile(diffs_mean, 0.025)),
        mean_diff_ci_high=float(np.nanquantile(diffs_mean, 0.975)),
        sharpe_diff=float(np.nanmean(diffs_sharpe)),
        sharpe_diff_p=p(diffs_sharpe),
        sharpe_diff_ci_low=float(np.nanquantile(diffs_sharpe, 0.025)),
        sharpe_diff_ci_high=float(np.nanquantile(diffs_sharpe, 0.975)),
        retvol_diff=float(np.nanmean(diffs_rvr)),
        retvol_diff_p=p(diffs_rvr),
        retvol_diff_ci_low=float(np.nanquantile(diffs_rvr, 0.025)),
        retvol_diff_ci_high=float(np.nanquantile(diffs_rvr, 0.975)),
    )

def build_ensemble_df(returns_by_method: dict) -> pd.DataFrame:
    ens = {LABEL_MAP.get(m, m): returns_by_method[m].median(axis=1) for m in returns_by_method}
    target = "SHAP Informed PPO Model"
    comps = [m for m in ens.keys() if m != target]
    rows = []
    for comp in comps:
        bb = bootstrap_ensemble_diffs(ens[target], ens[comp], block_len=21, B=2000, seed=42)
        rows.append(dict(Compare=f"{target} (median) vs {comp} (median)", **bb))
    return pd.DataFrame(rows)

def winrate_vs_ew_bar_from_df(pairwise_df: pd.DataFrame, universe_label: str, out_file: Path):
    metrics = ["CAGR", "Return/Vol", "Sharpe", "Calmar", "MaxDD", "Monthly_CVaR_5%"]
    wins, totals = [], []
    for m in metrics:
        row = pairwise_df[(pairwise_df["Compare"]=="SHAP Informed PPO Model vs Equal Weight") & (pairwise_df["Metric"]==m)]
        if row.empty:
            wins.append(np.nan); totals.append(np.nan)
        else:
            wins.append(float(row.iloc[0]["binom_k"])); totals.append(float(row.iloc[0]["binom_n"]))
    rates = [w/t if (t and not np.isnan(t) and t>0) else np.nan for w,t in zip(wins, totals)]
    plt.figure()
    xloc = np.arange(len(metrics))
    plt.bar(xloc, rates)
    plt.xticks(xloc, metrics, rotation=20, ha='right'); plt.ylim(0, 1.0)
    plt.xlabel("Metric"); plt.ylabel("Win rate vs Equal Weight (SHAP Informed PPO Model)")
    percentify_axis('y', decimals=0)
    for i, r in enumerate(rates):
        if not (r is None or np.isnan(r)):
            plt.text(i, min(0.98, r + 0.03), f"{int(wins[i])}/{int(totals[i])} ({r*100:.0f}%)", ha='center', va='bottom')
    finalize_fig(
        title=f"{universe_label}: Win Rate vs Equal Weight (30 runs)",
        caption="Win rate = fraction of SHAP Informed PPO runs beating Equal Weight on each metric.",
        legend_outside=False
    )
    savefig_both(out_file)

def forest_from_df(ensemble_df: pd.DataFrame, universe_label: str, out_prefix: Path):
    def clean_comp(s: str) -> str:
        if "vs " in s:
            right = s.split("vs", 1)[1]
            right = right.replace("(median)", "").strip()
            right = LABEL_MAP.get(right, right)
            return right
        return s
    df = ensemble_df.copy(); df["Comparator"] = df["Compare"].apply(clean_comp)
    for key, (diff_col, lo_col, hi_col, title_metric, xlabel) in {
        "mean": ("mean_diff","mean_diff_ci_low","mean_diff_ci_high","Mean Return","ΔMean Return"),
        "sharpe": ("sharpe_diff","sharpe_diff_ci_low","sharpe_diff_ci_high","Sharpe","ΔSharpe"),
        "retvol": ("retvol_diff","retvol_diff_ci_low","retvol_diff_ci_high","Return/Vol","ΔReturn/Vol"),
    }.items():
        y = np.arange(len(df))[::-1]
        diffs, lo, hi = df[diff_col].values, df[lo_col].values, df[hi_col].values
        err_low, err_high = diffs - lo, hi - diffs
        plt.figure()
        plt.errorbar(diffs, y, xerr=[err_low, err_high], fmt='o')
        plt.axvline(0.0, linestyle='--')
        plt.yticks(y, [wrap_label(c) for c in df["Comparator"].tolist()[::-1]])
        plt.xlabel(f"{xlabel} (SHAP Informed PPO − comparator)"); plt.ylabel("Comparator (median ensemble)")
        finalize_fig(
            title=f"{universe_label}: Ensemble Bootstrap — {title_metric} Differences",
            caption="Points show bootstrap mean differences; error bars are 95% percentile CIs (21-day block bootstrap).",
            legend_outside=False
        )
        savefig_both(out_prefix.with_name(out_prefix.stem + f"_forest_{key}.png"))


def make_universe_figures(excel_path: Path, out_dir: Path, universe_label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(excel_path)
    sheet_a = best_match(xls.sheet_names, ["shap+mom+stage1","shap mom stage1","shap_mom_stage1"])
    sheet_b = best_match(xls.sheet_names, ["mom+stage1","momentum stage1","mom_stage1"])
    sheet_c = best_match(xls.sheet_names, ["MeanCVaR","Mean CVaR","meancvar"])
    sheet_d = best_match(xls.sheet_names, ["Equal Weight","EqualWeight","EW"])

    mapping = {"SHAP+MOM+Stage1": sheet_a, "MOM+Stage1": sheet_b, "MeanCVaR": sheet_c, "Equal Weight": sheet_d}
    returns_by_method = {m: load_returns_df(xls, s) for m,s in mapping.items()}
    metrics_by_method = {m: compute_metrics_for_df(df) for m,df in returns_by_method.items()}
    ensemble_by_method = {m: returns_by_method[m].median(axis=1) for m in returns_by_method}

    cum_returns_plot(ensemble_by_method, universe_label, out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_cum_returns_median_ensemble.png")
    rolling_sharpe_plot(ensemble_by_method, universe_label, window=63, out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_rolling_sharpe_63.png")

    boxplot_metric(metrics_by_method, "CAGR", universe_label, "Annualized Return", out_dir / f"{universe_label.lower().replace(' ','_')}_box_CAGR.png", percent_axis=True)
    boxplot_metric(metrics_by_method, "Sharpe", universe_label, "Sharpe (annualized)", out_dir / f"{universe_label.lower().replace(' ','_')}_box_Sharpe.png", percent_axis=False)
    boxplot_metric(metrics_by_method, "Return/Vol", universe_label, "CAGR / Ann. Vol", out_dir / f"{universe_label.lower().replace(' ','_')}_box_ReturnVol.png", percent_axis=False)
    boxplot_metric(metrics_by_method, "Calmar", universe_label, "Calmar", out_dir / f"{universe_label.lower().replace(' ','_')}_box_Calmar.png", percent_axis=False)
    boxplot_metric(metrics_by_method, "MaxDD", universe_label, "Max Drawdown", out_dir / f"{universe_label.lower().replace(' ','_')}_box_MaxDD.png", percent_axis=True)
    boxplot_metric(metrics_by_method, "Monthly_CVaR_5%", universe_label, "Monthly CVaR (5%)", out_dir / f"{universe_label.lower().replace(' ','_')}_box_Monthly_CVaR_5pct.png", percent_axis=True)

    bar_means_ci(metrics_by_method, "CAGR", universe_label, "Annualized Return", out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_CAGR.png", percent_axis=True)
    bar_means_ci(metrics_by_method, "Sharpe", universe_label, "Sharpe (annualized)", out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_Sharpe.png", percent_axis=False)
    bar_means_ci(metrics_by_method, "Return/Vol", universe_label, "CAGR / Ann. Vol", out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_ReturnVol.png", percent_axis=False)
    bar_means_ci(metrics_by_method, "Calmar", universe_label, "Calmar", out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_Calmar.png", percent_axis=False)
    bar_means_ci(metrics_by_method, "MaxDD", universe_label, "Max Drawdown", out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_MaxDD.png", percent_axis=True)
    bar_means_ci(metrics_by_method, "Monthly_CVaR_5%", universe_label, "Monthly CVaR (5%)", out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_Monthly_CVaR_5pct.png", percent_axis=True)

    scatter_tradeoff(metrics_by_method, "AnnVol", "CAGR", universe_label, "Annualized Volatility", "CAGR", out_dir / f"{universe_label.lower().replace(' ','_')}_scatter_risk_return.png", x_percent=True, y_percent=True)
    scatter_tradeoff(metrics_by_method, "MaxDD", "Sharpe", universe_label, "Max Drawdown (more negative is worse)", "Sharpe (annualized)", out_dir / f"{universe_label.lower().replace(' ','_')}_scatter_sharpe_vs_maxdd.png", x_percent=True, y_percent=False)
    scatter_tradeoff(metrics_by_method, "Monthly_CVaR_5%", "CAGR", universe_label, "Monthly CVaR 5% (less negative is better)", "CAGR", out_dir / f"{universe_label.lower().replace(' ','_')}_scatter_cagr_vs_monthly_cvar.png", x_percent=True, y_percent=True)

    pairwise_df = build_pairwise_df(metrics_by_method)
    pairwise_df.to_csv(out_dir / f"{universe_label.lower().replace(' ','_')}_pairwise_tests_displaynames.csv", index=False)
    winrate_vs_ew_bar_from_df(pairwise_df, universe_label, out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_winrate_vs_ew.png")

    ensemble_df = build_ensemble_df(returns_by_method)
    ensemble_df.to_csv(out_dir / f"{universe_label.lower().replace(' ','_')}_ensemble_bootstrap_displaynames.csv", index=False)
    forest_from_df(ensemble_df, universe_label, out_prefix=out_dir / f"{universe_label.lower().replace(' ','_')}")

    return returns_by_method, metrics_by_method


# ========================= Turnover / HHI =============================
def load_daily_returns_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", axis=1)
    return df

def detect_iter_col(columns: list[str]) -> str | None:
    for k in columns:
        nk = k.lower()
        if nk in ("run", "iteration", "iter", "iter_id", "run_id"):
            return k
    return None

def load_weights_csv(path: Path, returns_cols: list[str]) -> pd.DataFrame:
    """
    Load portfolio weights CSV and return a tidy DataFrame indexed by Date with an 'Iter' column.
    Rows within each iteration are normalized to sum to 1 across tickers.
    """
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path}: missing 'Date' column.")
    it_col = detect_iter_col(list(df.columns))
    if it_col is None:
        raise ValueError(f"{path}: could not find iteration/run column. Add 'Run' or 'Iteration'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["Iter"] = df[it_col].astype(str)
    tickers = [c for c in df.columns if c in returns_cols]
    if not tickers:
        raise ValueError(f"{path}: no ticker columns match daily returns.")
    keep_cols = ["Date", "Iter"] + tickers
    df = df[keep_cols].copy()
    def _norm_row(g):
        w = g[tickers].astype(float)
        w = w.div(w.sum(axis=1), axis=0)
        g[tickers] = w
        return g
    df = df.groupby("Iter", group_keys=False).apply(_norm_row)
    df = df.set_index("Date").sort_index()
    return df

def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index=index, dtype=float).resample("M").last().index.intersection(index)

@dataclass
class SimResult:
    weights: pd.DataFrame
    port_daily: pd.Series
    turnover: pd.Series
    hhi_daily: pd.Series
    hhi_norm_daily: pd.Series

def _determine_sim_index(reb_dates: pd.DatetimeIndex, ret_idx: pd.DatetimeIndex, horizon_days: int = 21):
    first_reb, last_reb = reb_dates.min(), reb_dates.max()
    i0 = ret_idx.searchsorted(first_reb, side="left")
    i1 = ret_idx.searchsorted(last_reb, side="left")
    start = ret_idx[min(i0, len(ret_idx)-1)]
    end   = ret_idx[min(i1 + horizon_days, len(ret_idx)-1)]
    return ret_idx[ret_idx.searchsorted(start):ret_idx.searchsorted(end)+1]

def simulate_single_run(target_w: pd.DataFrame, daily_ret: pd.DataFrame) -> SimResult:
    tickers = list(target_w.columns)
    ret_idx = daily_ret.index
    sim_index = _determine_sim_index(target_w.index, ret_idx, horizon_days=21)
    w_plus   = pd.DataFrame(index=sim_index, columns=tickers, dtype=float)
    turnover = pd.Series(index=target_w.index.intersection(sim_index), dtype=float)
    port     = pd.Series(index=sim_index, dtype=float)
    first_day = sim_index[0]
    if first_day in target_w.index:
        w_plus.loc[first_day] = target_w.loc[first_day].values
    else:
        prev_t = target_w.loc[:first_day]
        w_plus.loc[first_day] = prev_t.iloc[-1].values
    port.loc[first_day] = float(np.dot(w_plus.loc[first_day].values, daily_ret.loc[first_day, tickers].values))
    for t in sim_index[1:]:
        prev = sim_index[sim_index.get_loc(t)-1]
        w_minus = w_plus.loc[prev].values * (1.0 + daily_ret.loc[t, tickers].values)
        s = w_minus.sum(); w_minus = (w_minus/s) if s!=0 else w_plus.loc[prev].values.copy()
        if t in target_w.index:
            target = target_w.loc[t].values
            turnover.loc[t] = float(0.5 * np.abs(target - w_minus).sum())
            w_plus.loc[t] = target
        else:
            w_plus.loc[t] = w_minus
        port.loc[t] = float(np.dot(w_plus.loc[prev].values, daily_ret.loc[t, tickers].values))
    hhi = (w_plus**2).sum(axis=1)
    n = len(tickers)
    hhi_norm = (hhi - (1.0/n)) / (1.0 - (1.0/n))
    return SimResult(weights=w_plus, port_daily=port, turnover=turnover, hhi_daily=hhi, hhi_norm_daily=hhi_norm)

def simulate_equal_weight(daily_ret: pd.DataFrame, sim_index: pd.DatetimeIndex, rebalance_dates: pd.DatetimeIndex) -> SimResult:
    tickers = list(daily_ret.columns)
    n = len(tickers); ew = np.repeat(1.0/n, n)
    w_plus   = pd.DataFrame(index=sim_index, columns=tickers, dtype=float)
    turnover = pd.Series(index=rebalance_dates.intersection(sim_index), dtype=float)
    port     = pd.Series(index=sim_index, dtype=float)
    first_day = sim_index[0]
    w_plus.loc[first_day] = ew
    port.loc[first_day] = float(np.dot(ew, daily_ret.loc[first_day, tickers].values))
    for t in sim_index[1:]:
        prev = sim_index[sim_index.get_loc(t)-1]
        w_minus = w_plus.loc[prev].values * (1.0 + daily_ret.loc[t, tickers].values)
        s = w_minus.sum(); w_minus = (w_minus/s) if s!=0 else w_plus.loc[prev].values.copy()
        if t in rebalance_dates:
            turnover.loc[t] = float(0.5 * np.abs(ew - w_minus).sum())
            w_plus.loc[t] = ew
        else:
            w_plus.loc[t] = w_minus
        port.loc[t] = float(np.dot(w_plus.loc[prev].values, daily_ret.loc[t, tickers].values))
    hhi = (w_plus**2).sum(axis=1); hhi_norm = (hhi - (1.0/n)) / (1.0 - (1.0/n))
    return SimResult(weights=w_plus, port_daily=port, turnover=turnover, hhi_daily=hhi, hhi_norm_daily=hhi_norm)

@dataclass
class RunSummary:
    iteration: str
    ann_return: float
    ann_vol: float
    sharpe: float
    maxdd: float
    m_cvar5: float
    ann_turnover: float
    mean_hhi_norm: float

def annualized_turnover(turnover_series: pd.Series, sim_index: pd.DatetimeIndex) -> float:
    t = turnover_series.dropna()
    if len(t) == 0: return 0.0
    years = len(sim_index) / TRADING_DAYS
    rebs_per_year = (len(t) / years) if years > 0 else np.nan
    return float(t.mean() * rebs_per_year) if pd.notnull(rebs_per_year) else np.nan

def compute_monthly_cvar5(dr: pd.Series) -> float:
    m = block_monthly_returns(dr, 21)
    if len(m) == 0: return np.nan
    return float(m[m <= m.quantile(0.05)].mean())

def summarize_run(sim: SimResult, iteration: str, full_index: pd.DatetimeIndex) -> RunSummary:
    dr = sim.port_daily.reindex(full_index).dropna()
    if dr.empty:
        return RunSummary(iteration, *(np.nan,)*7)
    ann_ret = compute_cagr(dr)
    ann_vol = compute_ann_vol(dr)
    sharpe  = compute_sharpe(dr)
    maxdd   = compute_max_drawdown(dr)
    cvar5   = compute_monthly_cvar5(dr=dr)
    ann_to  = annualized_turnover(sim.turnover, full_index)
    hhi     = sim.hhi_norm_daily.reindex(full_index).ffill().fillna(0.0)
    mean_hhi = float(hhi.mean()) if len(hhi) else np.nan
    return RunSummary(iteration, ann_ret, ann_vol, sharpe, maxdd, cvar5, ann_to, mean_hhi)

def run_model(weights_all: pd.DataFrame, daily_ret: pd.DataFrame):
    tickers = [c for c in weights_all.columns if c not in ("Iter")]
    iters = sorted(weights_all["Iter"].unique(), key=lambda x: int(re.findall(r'\\d+', x)[0]) if re.findall(r'\\d+', x) else int(x) if str(x).isdigit() else x)
    all_reb = pd.Index([])
    for it in iters:
        all_reb = all_reb.union(weights_all[weights_all["Iter"]==it].index)
    sim_index = _determine_sim_index(all_reb, daily_ret.index, horizon_days=21)
    returns_by_iter, hhi_by_iter, summaries = {}, {}, []
    for it in iters:
        w_it = weights_all[weights_all["Iter"]==it][tickers].copy()
        sim  = simulate_single_run(w_it, daily_ret[tickers])
        ssum = summarize_run(sim, iteration=str(it), full_index=sim_index)
        summaries.append(ssum)
        returns_by_iter[str(it)] = sim.port_daily.reindex(sim_index).fillna(0.0)
        hhi_by_iter[str(it)]     = sim.hhi_norm_daily.reindex(sim_index).ffill().fillna(0.0)
    metrics_by_iter = pd.DataFrame({
        "Iteration": [s.iteration for s in summaries],
        "Annualized Return": [s.ann_return for s in summaries],
        "Annualized Volatility": [s.ann_vol for s in summaries],
        "Sharpe (rf≈0)": [s.sharpe for s in summaries],
        "Max Drawdown": [s.maxdd for s in summaries],
        "Monthly CVaR(5%)": [s.m_cvar5 for s in summaries],
        "Annualized Turnover (one-way)": [s.ann_turnover for s in summaries],
        "Mean HHI (normalized)": [s.mean_hhi_norm for s in summaries],
    }).set_index("Iteration").sort_index()
    df_ret = pd.DataFrame(returns_by_iter); ensemble = df_ret.median(axis=1)
    df_hhi = pd.DataFrame(hhi_by_iter); hhi_median = df_hhi.median(axis=1)
    return metrics_by_iter, returns_by_iter, hhi_by_iter, ensemble, hhi_median, sim_index

def build_ew_like(weights_all: pd.DataFrame, daily_ret: pd.DataFrame):
    iters = sorted(weights_all["Iter"].unique(), key=lambda x: int(re.findall(r'\\d+', x)[0]) if re.findall(r'\\d+', x) else int(x) if str(x).isdigit() else x)
    all_reb = pd.Index([])
    for it in iters: all_reb = all_reb.union(weights_all[weights_all["Iter"]==it].index)
    sim_index = _determine_sim_index(all_reb, daily_ret.index, horizon_days=21)
    returns_by_iter, hhi_by_iter, summaries = {}, {}, []
    for it in iters:
        reb = weights_all[weights_all["Iter"]==it].index
        sim = simulate_equal_weight(daily_ret, sim_index, reb)
        ssum = summarize_run(sim, iteration=str(it), full_index=sim_index)
        summaries.append(ssum)
        returns_by_iter[str(it)] = sim.port_daily.reindex(sim_index).fillna(0.0)
        hhi_by_iter[str(it)]     = sim.hhi_norm_daily.reindex(sim_index).ffill().fillna(0.0)
    metrics = pd.DataFrame({
        "Iteration": [s.iteration for s in summaries],
        "Annualized Return": [s.ann_return for s in summaries],
        "Annualized Volatility": [s.ann_vol for s in summaries],
        "Sharpe (rf≈0)": [s.sharpe for s in summaries],
        "Max Drawdown": [s.maxdd for s in summaries],
        "Monthly CVaR(5%)": [s.m_cvar5 for s in summaries],
        "Annualized Turnover (one-way)": [s.ann_turnover for s in summaries],
        "Mean HHI (normalized)": [s.mean_hhi_norm for s in summaries],
    }).set_index("Iteration").sort_index()
    df_ret = pd.DataFrame(returns_by_iter); ensemble = df_ret.median(axis=1)
    df_hhi = pd.DataFrame(hhi_by_iter); hhi_median = df_hhi.median(axis=1)
    return metrics, returns_by_iter, hhi_by_iter, ensemble, hhi_median


def bar_means_ci_series(values_by_model: dict[str, pd.Series], title: str, ylabel: str, out_file: Path, as_percent=False):
    plt.figure()
    models = list(values_by_model.keys())
    means, cis = [], []
    for m in models:
        x = values_by_model[m].dropna().values
        if len(x) == 0:
            means.append(np.nan); cis.append(0.0)
        else:
            mean = float(np.mean(x))
            se   = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x)>1 else 0.0
            ci   = 1.96 * se
            means.append(mean); cis.append(ci)
    xpos = np.arange(len(models))
    plt.bar(xpos, means, yerr=cis, capsize=3)
    plt.xticks(xpos, [DISPLAY_MAP.get(m, m) for m in models], rotation=0)
    plt.xlabel("Model"); plt.ylabel(ylabel)
    if as_percent:
        percentify_axis('y', decimals=1)
        for i, m in enumerate(means):
            if not (m is None or np.isnan(m)):
                plt.text(i, m + (cis[i] if i < len(cis) else 0) + 0.004, f"{m*100:.1f}%", ha='center', va='bottom', fontsize=9)
    finalize_fig(title=title, caption="Bars show mean across runs; error bars are normal-approximate 95% CIs.", legend_outside=False)
    savefig_both(out_file)

def boxplot_by_model(values_by_model: dict[str, pd.Series], title: str, ylabel: str, out_file: Path, as_percent=False):
    plt.figure()
    models = list(values_by_model.keys())
    data = [values_by_model[m].dropna().values for m in models]
    labels = [wrap_label(DISPLAY_MAP.get(m, m)).replace(" ", "\\n") for m in models]
    try:
        plt.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=labels, showmeans=True)
    plt.xlabel("Model"); plt.ylabel(ylabel)
    if as_percent: percentify_axis('y', decimals=1)
    finalize_fig(title=title, caption="Boxes: IQR; center line: median; diamond = mean; whiskers: non-outlier range.", legend_outside=False)
    savefig_both(out_file)

def hhi_timeseries_plot(hhi_median_by_model: dict[str, pd.Series], title: str, out_file: Path):
    plt.figure()
    ax = plt.gca()
    for m, s in hhi_median_by_model.items():
        s = s.dropna()
        if s.empty: continue
        disp = DISPLAY_MAP.get(m, m)
        ax.plot(s.index, s.values, label=disp, linewidth=1.6)
    ax.set_xlabel("Date"); ax.set_ylabel("Median HHI (normalized)")
    loc = AutoDateLocator(); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    finalize_fig(title=title, caption="Median across runs of daily normalized HHI (0=equi‑weight, 1=fully concentrated).", legend_outside=False)
    savefig_both(out_file)


def run_all(
    excel_path: Path,
    daily_returns_path: Path,
    shap_weights_path: Path,
    ppo_weights_path: Path,
    meancvar_weights_path: Path,
    out_dir: Path,
    universe_label: str = "Sector ETFs",
    schedule_from: str = "SHAP"
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    make_universe_figures(excel_path=excel_path, out_dir=out_dir, universe_label=universe_label)

    ret = load_daily_returns_csv(daily_returns_path); tickers = ret.columns.tolist()
    shap_w = load_weights_csv(shap_weights_path, tickers)
    ppo_w  = load_weights_csv(ppo_weights_path,  tickers)
    mc_w   = load_weights_csv(meancvar_weights_path, tickers)

    m_shap = run_model(shap_w, ret)
    m_ppo  = run_model(ppo_w,  ret)
    m_mc   = run_model(mc_w,   ret)
    sched_map = {"SHAP": shap_w, "PPO": ppo_w, "MeanCVaR": mc_w}
    ew      = build_ew_like(sched_map.get(schedule_from, shap_w), ret)

    m_shap[0].to_csv(out_dir / "metrics_by_iter_SHAP.csv")
    m_ppo[0].to_csv(out_dir / "metrics_by_iter_PPO.csv")
    m_mc[0].to_csv(out_dir / "metrics_by_iter_MeanCVaR.csv")
    ew[0].to_csv(out_dir / f"metrics_by_iter_EqualWeight_{schedule_from}.csv")

    bar_means_ci_series(
        values_by_model={
            "SHAP": m_shap[0]["Annualized Turnover (one-way)"],
            "PPO":  m_ppo[0]["Annualized Turnover (one-way)"],
            "MeanCVaR": m_mc[0]["Annualized Turnover (one-way)"],
            "EqualWeight": ew[0]["Annualized Turnover (one-way)"],
        },
        title=f"{universe_label}: Annualized Turnover (one‑way) — Mean ± 95% CI",
        ylabel="Annualized Turnover (one‑way)",
        out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_turnover.png",
        as_percent=False
    )
    bar_means_ci_series(
        values_by_model={
            "SHAP": m_shap[0]["Mean HHI (normalized)"],
            "PPO":  m_ppo[0]["Mean HHI (normalized)"],
            "MeanCVaR": m_mc[0]["Mean HHI (normalized)"],
            "EqualWeight": ew[0]["Mean HHI (normalized)"],
        },
        title=f"{universe_label}: Concentration (HHI normalized) — Mean ± 95% CI",
        ylabel="Mean HHI (normalized)",
        out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_bar_meanci_hhi.png",
        as_percent=False
    )
    boxplot_by_model(
        {
            "SHAP": m_shap[0]["Annualized Turnover (one-way)"],
            "PPO":  m_ppo[0]["Annualized Turnover (one-way)"],
            "MeanCVaR": m_mc[0]["Annualized Turnover (one-way)"],
            "EqualWeight": ew[0]["Annualized Turnover (one-way)"],
        },
        title=f"{universe_label}: Annualized Turnover (one‑way) — Distribution Across Runs",
        ylabel="Annualized Turnover (one‑way)",
        out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_box_turnover.png",
        as_percent=False
    )
    boxplot_by_model(
        {
            "SHAP": m_shap[0]["Mean HHI (normalized)"],
            "PPO":  m_ppo[0]["Mean HHI (normalized)"],
            "MeanCVaR": m_mc[0]["Mean HHI (normalized)"],
            "EqualWeight": ew[0]["Mean HHI (normalized)"],
        },
        title=f"{universe_label}: Concentration (HHI normalized) — Distribution Across Runs",
        ylabel="Mean HHI (normalized)",
        out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_box_hhi.png",
        as_percent=False
    )
    hhi_timeseries_plot(
        {
            "SHAP": m_shap[4],
            "PPO":  m_ppo[4],
            "MeanCVaR": m_mc[4],
            "EqualWeight": ew[4],
        },
        title=f"{universe_label}: Median Normalized HHI Over Time",
        out_file=out_dir / f"{universe_label.lower().replace(' ','_')}_hhi_timeseries_median.png"
    )
    return True


def run_tests():
    print("Running basic tests for turnover & HHI...")
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    ret = pd.DataFrame({"A":[0.0,0.0,0.0], "B":[0.0,0.0,0.0]}, index=dates)
    w = pd.DataFrame({
        "Date": [dates[0], dates[2]],
        "Iter": ["1","1"],
        "A": [1.0, 0.0],
        "B": [0.0, 1.0]
    }).set_index("Date")
    sim = simulate_single_run(w[["A","B"]], ret[["A","B"]])
    to = sim.turnover.dropna()
    assert np.isclose(to.iloc[-1], 1.0, atol=1e-10), f"Expected turnover 1.0, got {to.iloc[-1]}"
    print("  ✓ Turnover full-switch test passed.")
    n=4
    w_eq = pd.Series(np.repeat(1.0/n,n))
    hhi_eq = float((w_eq**2).sum()); hhi_eq_norm = (hhi_eq - (1.0/n)) / (1.0 - (1.0/n))
    assert np.isclose(hhi_eq_norm, 0.0, atol=1e-12), f"Expected 0.0, got {hhi_eq_norm}"
    w_fc = pd.Series([1.0,0,0,0])
    hhi_fc = float((w_fc**2).sum()); hhi_fc_norm = (hhi_fc - (1.0/n)) / (1.0 - (1.0/n))
    assert np.isclose(hhi_fc_norm, 1.0, atol=1e-12), f"Expected 1.0, got {hhi_fc_norm}"
    print("  ✓ HHI normalization test passed.")
    print("All tests passed.")


def _cli():
    ap = argparse.ArgumentParser(description="PPO analysis: packet + turnover/HHI")
    ap.add_argument("--workbook", type=str, help="Excel workbook for packet charts.")
    ap.add_argument("--out", type=str, default="./figs_out", help="Output directory.")
    ap.add_argument("--universe-label", type=str, default="Sector ETFs")
    ap.add_argument("--daily-returns", type=str, help="CSV of daily returns (Date + tickers).")
    ap.add_argument("--weights-shap", type=str, help="CSV of SHAP-informed PPO weights.")
    ap.add_argument("--weights-ppo", type=str, help="CSV of PPO weights.")
    ap.add_argument("--weights-meanc", type=str, help="CSV of Mean-CVaR(+ penalty) weights.")
    ap.add_argument("--schedule-from", type=str, default="SHAP", choices=["SHAP","PPO","MeanCVaR"])
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    if args.run_tests:
        run_tests(); return

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.workbook:
        make_universe_figures(Path(args.workbook), out_dir, universe_label=args.universe_label)

    if all([args.daily_returns, args.weights_shap, args.weights_ppo, args.weights_meanc]):
        run_all(
            excel_path=Path(args.workbook) if args.workbook else Path(args.daily_returns),
            daily_returns_path=Path(args.daily_returns),
            shap_weights_path=Path(args.weights_shap),
            ppo_weights_path=Path(args.weights_ppo),
            meancvar_weights_path=Path(args.weights_meanc),
            out_dir=out_dir,
            universe_label=args.universe_label,
            schedule_from=args.schedule_from
        )
    print(f"Done. Outputs at: {out_dir.as_posix()}")

if __name__ == "__main__":
    _cli()
