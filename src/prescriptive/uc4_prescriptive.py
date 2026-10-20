"""
UC4 – Prescriptive analytics: allocating high-dependency unit (HDU) beds to patients
admitted with sepsis.

PROHI HT26, Group 2. Owner: Qizheng (Kevin) Zhao.

Intended user
    The hospital bed manager, who decides each day which newly admitted sepsis patients
    are placed in the HDU when the HDU has fewer free beds than there are candidates.

Questions
    Q1  Above which predicted risk of in-hospital death is closer monitoring worthwhile?
        (decision curve analysis on cross-fitted predictions)
    Q2  When HDU beds are scarce, which allocation rule places the most patients who would
        otherwise die in the HDU, and what does an equity constraint cost?
        (discrete-event simulation with a constraint-programming allocation model)

Pipeline
    1. Risk model   logistic regression on age (cubic spline), sex and episode number,
                    fitted on the Sepsis-3 study cohort; out-of-fold predictions from
                    5-fold cross-fitting so that no patient is scored by a model that saw it.
    2. Validation   AUC with bootstrap CI, calibration intercept and slope, decile plot,
                    external check on the South Korean cohort.
    3. Thresholds   net benefit vs. "monitor all" and "monitor none", bootstrap bands.
    4. Allocation   90-day simulation of a 6-bed HDU (two single-sex bays of three beds,
                    length of stay drawn per patient). Each day the bed manager allocates
                    the free beds among the day's admissions. Six policies are compared on
                    the same simulated days (common random numbers):
                      FCFS            first come, first served (current practice)
                      Threshold       FCFS among patients with predicted risk >= 20 %
                      Risk ranking    highest predicted risk first, bed-by-bed
                      CP-SAT          optimal daily allocation (OR-Tools), bay constraints
                      CP-SAT + equity at least 15 % of HDU placements over the period to patients < 65 y
                      Oracle          knows who will die; an upper bound, not a policy
    5. Evaluation   outcome-based: the number of patients placed in the HDU who in fact died
                    in hospital, per 1,000 admissions. This avoids scoring the model with its
                    own probabilities. Deaths averted = RRR x that number, where RRR (the
                    relative risk reduction from HDU care) is an assumption; RRR only rescales
                    the results and does not change the ranking of the policies.

Run
    python uc4_prescriptive.py --data-dir data --out-dir results/uc4
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

SEED = 2026
# File names as stored in the team repository (resources/sepsis_survival_dataset/).
# Originally the UCI names: s41598-020-73558-3_sepsis_survival_{primary,study,validation}_cohort.csv
FILES = {
    "primary": "1_primary_cohort.csv",
    "study": "3_study_cohort.csv",
    "validation": "2_validation_cohort.csv",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "resources" / "sepsis_survival_dataset"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "uc4"


# --------------------------------------------------------------------------- data
def load_cohort(data_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / FILES[name])
    df.columns = ["age", "sex", "episode", "alive"]
    df["died"] = 1 - df["alive"]
    df["episode_cat"] = df["episode"].clip(upper=3)          # 1, 2, >=3 (few patients have 4-5)
    return df


# --------------------------------------------------------------------------- step 1: risk model
def make_model() -> Pipeline:
    pre = ColumnTransformer([
        ("age", SplineTransformer(n_knots=4, degree=3, include_bias=False), ["age"]),
        ("sex", "passthrough", ["sex"]),
        ("episode", OneHotEncoder(drop="first", categories=[[1, 2, 3]]), ["episode_cat"]),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1e6, max_iter=2000))])


FEATURES = ["age", "sex", "episode_cat"]


def cross_fitted_risk(df: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Out-of-fold predicted risk for every admission."""
    p = np.zeros(len(df))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(df, df["died"]):
        m = make_model().fit(df.iloc[tr][FEATURES], df.iloc[tr]["died"])
        p[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    return p


def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict:
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(lp, y).coef_[0][0]
    # calibration-in-the-large: intercept with logit(p) as offset (one Newton fit)
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp.ravel())))
        a += np.sum(y - mu) / np.sum(mu * (1 - mu))
    return {"calibration_intercept": float(a), "calibration_slope": float(slope),
            "observed_mortality": float(y.mean()), "mean_predicted": float(p.mean())}


def bootstrap_auc(y, p, n=1000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def decile_calibration(y, p, bins=10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    d["se"] = np.sqrt(d.obs * (1 - d.obs) / d.n)
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- step 3: decision curve
def net_benefit(y, p, thresholds):
    n, prev = len(y), y.mean()
    nb_model, nb_all = [], []
    for t in thresholds:
        flag = p >= t; w = t / (1 - t)
        tp = np.sum(flag & (y == 1)); fp = np.sum(flag & (y == 0))
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prev - (1 - prev) * w)
    return np.array(nb_model), np.array(nb_all)


def decision_curve(y, p, thresholds, n_boot=500, rng=None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(SEED)
    m, a = net_benefit(y, p, thresholds)
    boots_m, boots_d = [], []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        bm, ba = net_benefit(y[b], p[b], thresholds)
        boots_m.append(bm); boots_d.append(bm - np.maximum(ba, 0))
    bm, bd = np.array(boots_m), np.array(boots_d)
    return pd.DataFrame({"threshold": thresholds, "nb_model": m, "nb_all": a, "nb_none": 0.0,
                         "nb_model_lo": np.percentile(bm, 2.5, 0), "nb_model_hi": np.percentile(bm, 97.5, 0),
                         "delta_vs_best_default": m - np.maximum(a, 0),
                         "delta_lo": np.percentile(bd, 2.5, 0), "delta_hi": np.percentile(bd, 97.5, 0)})


def threshold_table(y, p, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        f = p >= t; tp = np.sum(f & (y == 1)); fp = np.sum(f & (y == 0)); fn = np.sum(~f & (y == 1))
        rows.append({"Threshold": f"{t:.0%}", "Flagged per 1,000": int(round(1000 * f.mean())),
                     "Deaths captured (%)": round(100 * tp / (tp + fn), 1),
                     "PPV (%)": round(100 * tp / max(tp + fp, 1), 1),
                     "False alarms per death captured": round(fp / max(tp, 1), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- step 4: allocation
@dataclass
class Ward:
    bays: int = 2
    beds_per_bay: int = 3
    single_sex_bays: bool = True


def _bay_state(occupants, day, ward):
    """free beds and current sex (None if empty) for each bay."""
    free, sex = [], []
    for b in range(ward.bays):
        cur = [o for o in occupants if o["bay"] == b and o["out"] > day]
        free.append(ward.beds_per_bay - len(cur))
        sex.append(cur[0]["sex"] if cur else None)
    return free, sex


def _place_in_order(order, cand, free, sex, ward):
    """Bed-by-bed placement in a given patient order (how a bed manager works by hand).
    Prefers a bay already holding the same sex, so an empty bay stays open for either sex."""
    free, sex, chosen = list(free), list(sex), []
    for i in order:
        s = cand["sex"][i]
        options = [b for b in range(ward.bays) if free[b] > 0 and (not ward.single_sex_bays or sex[b] in (None, s))]
        if not options:
            continue
        options.sort(key=lambda b: (sex[b] is None, -free[b]))
        b = options[0]; chosen.append((i, b)); free[b] -= 1; sex[b] = s
    return chosen


def _cp_sat(cand, free, sex, ward, min_young_share=0.0, young_age=65, cum_young=0, cum_total=0):
    n = len(cand["p"])
    if n == 0 or sum(free) == 0:
        return []
    m = cp_model.CpModel()
    x = {}
    bay_female = {b: m.NewBoolVar(f"f{b}") for b in range(ward.bays) if sex[b] is None}
    for i in range(n):
        for b in range(ward.bays):
            if free[b] == 0:
                continue
            if ward.single_sex_bays and sex[b] is not None and sex[b] != cand["sex"][i]:
                continue
            x[i, b] = m.NewBoolVar(f"x{i}_{b}")
            if ward.single_sex_bays and sex[b] is None:
                m.Add(x[i, b] <= (bay_female[b] if cand["sex"][i] == 1 else 1 - bay_female[b]))
    for i in range(n):
        m.Add(sum(v for (j, _), v in x.items() if j == i) <= 1)
    for b in range(ward.bays):
        m.Add(sum(v for (_, c), v in x.items() if c == b) <= free[b])
    objective = sum(int(round(cand["p"][i] * 10_000)) * v for (i, _), v in x.items())
    young = [v for (i, _), v in x.items() if cand["age"][i] < young_age] if min_young_share > 0 else []
    if young:
        # cumulative over the simulated period: younger patients must hold at least
        # `min_young_share` of all HDU placements so far (integer form). Days without a
        # younger candidate impose nothing. If the quota cannot be met today, the shortfall
        # is minimised first (lexicographic penalty) instead of leaving beds empty.
        k = 100
        short = m.NewIntVar(0, k * (cum_total + len(cand["p"])), "shortfall")
        m.Add(k * (cum_young + sum(young)) + short >= int(round(k * min_young_share)) * (cum_total + sum(x.values())))
        objective = objective - 10_000_000 * short
    m.Maximize(objective)
    s = cp_model.CpSolver(); s.parameters.num_workers = 1; s.parameters.max_time_in_seconds = 2.0
    s.parameters.random_seed = SEED
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT returned no solution")
    return [(i, b) for (i, b), v in x.items() if s.Value(v)]


POLICIES = ["FCFS", "Threshold 20%", "Risk ranking", "CP-SAT", "CP-SAT + equity", "Oracle"]


def allocate(policy, cand, free, sex, ward, threshold=0.20, min_young_share=0.15, cum=(0, 0)):
    n = len(cand["p"])
    if policy == "FCFS":
        return _place_in_order(range(n), cand, free, sex, ward)
    if policy == "Threshold 20%":
        return _place_in_order([i for i in range(n) if cand["p"][i] >= threshold], cand, free, sex, ward)
    if policy == "Risk ranking":
        return _place_in_order(np.argsort(-cand["p"], kind="stable"), cand, free, sex, ward)
    if policy == "CP-SAT":
        return _cp_sat(cand, free, sex, ward)
    if policy == "CP-SAT + equity":
        return _cp_sat(cand, free, sex, ward, min_young_share=min_young_share, cum_young=cum[0], cum_total=cum[1])
    if policy == "Oracle":
        return _place_in_order(np.lexsort((-cand["p"], -cand["died"])), cand, free, sex, ward)
    raise ValueError(policy)


def simulate(pool: pd.DataFrame, policies, n_runs=200, days=90, arrivals_per_day=8.0,
             ward: Ward = Ward(), mean_los=3.0, seed=SEED) -> pd.DataFrame:
    """Paired simulation: every policy sees the same admissions and lengths of stay."""
    rng = np.random.default_rng(seed)
    P, S, A, D = (pool[c].to_numpy() for c in ("p", "sex", "age", "died"))
    rows = []
    for r in range(n_runs):
        n_day = rng.poisson(arrivals_per_day, size=days)
        draws = [rng.choice(len(pool), size=k, replace=True) for k in n_day]   # arrival order = draw order
        los = [1 + rng.poisson(mean_los - 1, size=k) for k in n_day]
        for pol in policies:
            occ, hdu, deaths, exp, young, bed_days = [], 0, 0, 0.0, 0, 0
            for d in range(days):
                idx = draws[d]
                cand = {"p": P[idx], "sex": S[idx], "age": A[idx], "died": D[idx]}
                free, sex = _bay_state(occ, d, ward)
                chosen = allocate(pol, cand, free, sex, ward, cum=(young, hdu))
                bed_days += ward.bays * ward.beds_per_bay - sum(free) + len(chosen)
                for i, b in chosen:
                    occ.append({"bay": b, "sex": cand["sex"][i], "out": d + int(los[d][i])})
                    hdu += 1; deaths += int(cand["died"][i]); exp += cand["p"][i]
                    young += int(cand["age"][i] < 65)
                occ = [o for o in occ if o["out"] > d]
            n_adm = int(n_day.sum())
            rows.append({"run": r, "policy": pol, "admissions": n_adm, "hdu_admissions": hdu,
                         "deaths_in_hdu": deaths, "expected_deaths_in_hdu": exp,
                         "under65_share": young / max(hdu, 1),
                         "occupancy": bed_days / (days * ward.bays * ward.beds_per_bay),
                         "deaths_in_hdu_per_1000": 1000 * deaths / n_adm})
    return pd.DataFrame(rows)


def paired_summary(sim: pd.DataFrame, rrr: float, ref="FCFS") -> pd.DataFrame:
    w = sim.pivot(index="run", columns="policy", values="deaths_in_hdu_per_1000")
    out = []
    for pol in [c for c in POLICIES if c in w.columns]:
        diff = w[pol] - w[ref]
        g = sim[sim.policy == pol]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        out.append({"policy": pol,
                    "deaths_in_HDU_per_1000": w[pol].mean(),
                    "vs_FCFS": diff.mean(), "vs_FCFS_lo": diff.mean() - 1.96 * se, "vs_FCFS_hi": diff.mean() + 1.96 * se,
                    "share_of_runs_better_than_FCFS": float((diff > 0).mean()),
                    "deaths_averted_per_1000_at_RRR": rrr * w[pol].mean(),
                    "extra_deaths_averted_vs_FCFS_per_1000": rrr * diff.mean(),
                    "HDU_mortality_%": 100 * g.deaths_in_hdu.sum() / g.hdu_admissions.sum(),
                    "under65_share_%": 100 * g.under65_share.mean(),
                    "occupancy_%": 100 * g.occupancy.mean()})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- main
def run(data_dir: Path, out_dir: Path, n_runs: int, rrr: float, quick: bool = False):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    study = load_cohort(data_dir, "study")
    korea = load_cohort(data_dir, "validation")

    # 1-2 risk model and its validation
    study["p"] = cross_fitted_risk(study)
    y, p = study["died"].to_numpy(), study["p"].to_numpy()
    from scipy.stats import spearmanr
    primary = load_cohort(data_dir, "primary")
    p_prim = cross_fitted_risk(primary)
    p_age = np.zeros(len(study))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(study, study["died"]):
        p_age[te] = LogisticRegression(C=1e6).fit(study.iloc[tr][["age"]], study.iloc[tr]["died"]).predict_proba(study.iloc[te][["age"]])[:, 1]
    lin = Pipeline([("lr", LogisticRegression(C=1e6, max_iter=2000))])
    p_lin = np.zeros(len(study))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(study, y):
        p_lin[te] = lin.fit(study.iloc[tr][["age", "sex", "episode"]], y[tr]).predict_proba(study.iloc[te][["age", "sex", "episode"]])[:, 1]
    final = make_model().fit(study[FEATURES], y)
    pk = final.predict_proba(korea[FEATURES])[:, 1]
    summary = {
        "n_study": int(len(study)), "deaths_study": int(y.sum()),
        "auc_crossfit": roc_auc_score(y, p), "auc_crossfit_ci": bootstrap_auc(y, p, 300 if quick else 1000, rng),
        "auc_all_linear_terms": roc_auc_score(y, p_lin),
        "auc_age_only": roc_auc_score(y, p_age),
        "spearman_risk_age": float(spearmanr(p, study["age"]).correlation),
        "auc_primary_cohort_crossfit": roc_auc_score(primary["died"], p_prim),
        "primary_under65_share": float((primary["age"] < 65).mean()),
        "study_under65_share": float((study["age"] < 65).mean()),
        "primary_mortality": float(primary["died"].mean()),
        "max_risk_under65": float(p[study["age"].to_numpy() < 65].max()),
        "korea_deaths": int(korea["died"].sum()),
        **calibration_stats(y, p),
        "n_korea": int(len(korea)), "auc_korea": roc_auc_score(korea["died"], pk),
        "korea_observed": float(korea["died"].mean()), "korea_mean_predicted": float(pk.mean()),
        "risk_p10_p50_p90": [float(v) for v in np.percentile(p, [10, 50, 90])],
    }
    decile_calibration(y, p).to_csv(out_dir / "calibration_deciles.csv", index=False)
    study[["age", "sex", "episode", "died", "p"]].to_csv(out_dir / "crossfitted_risk.csv", index=False)

    # 3 decision curve
    th = np.round(np.arange(0.05, 0.401, 0.01), 2)
    dca = decision_curve(y, p, th, 200 if quick else 500, rng)
    dca.to_csv(out_dir / "decision_curve.csv", index=False)
    # contiguous range from the lowest threshold where the bootstrap interval of the
    # difference (model minus best default) stays above zero
    ok = (dca.delta_lo > 0).to_numpy(); first = int(np.argmax(ok)); last = first
    while last + 1 < len(ok) and ok[last + 1]:
        last += 1
    sig = dca.threshold.iloc[first:last + 1] if ok.any() else dca.threshold.iloc[0:0]
    summary["dca_max_gain"] = float(dca.delta_vs_best_default.max())
    summary["dca_max_gain_at"] = float(dca.threshold.iloc[int(dca.delta_vs_best_default.argmax())])
    summary["dca_point_range"] = [float(dca[dca.delta_vs_best_default > 0.0005].threshold.min()),
                                  float(dca[dca.delta_vs_best_default > 0.0005].threshold.max())]
    summary["dca_ci_range"] = [float(sig.min()), float(sig.max())] if len(sig) else None
    threshold_table(y, p, [0.10, 0.15, 0.20, 0.25]).to_csv(out_dir / "threshold_table.csv", index=False)

    # 4-5 allocation
    sim = simulate(study, POLICIES, n_runs=n_runs)
    sim.to_csv(out_dir / "simulation_runs.csv", index=False)
    ps = paired_summary(sim, rrr)
    ps.round(3).to_csv(out_dir / "policy_summary.csv", index=False)

    sens = []
    for beds in (2, 3, 4):              # beds per bay -> 4, 6, 8 HDU beds
        for lam in (6, 8, 12):
            s = simulate(study, ["FCFS", "CP-SAT"], n_runs=max(20, n_runs // 5), arrivals_per_day=lam,
                         ward=Ward(beds_per_bay=beds), seed=SEED + beds * 100 + lam)
            w = s.pivot(index="run", columns="policy", values="deaths_in_hdu_per_1000")
            occ = s[s.policy == "CP-SAT"].occupancy.mean()
            sens.append({"hdu_beds": 2 * beds, "admissions_per_day": lam,
                         "cp_minus_fcfs_per_1000": (w["CP-SAT"] - w["FCFS"]).mean(),
                         "fcfs_per_1000": w["FCFS"].mean(), "cp_per_1000": w["CP-SAT"].mean(), "occupancy": occ})
    pd.DataFrame(sens).round(3).to_csv(out_dir / "sensitivity_capacity_demand.csv", index=False)

    # bay constraint on/off: how much does single-sex bay management cost?
    # paired with the main run: same seed and number of runs gives the same admissions
    s2 = simulate(study, ["CP-SAT"], n_runs=n_runs, ward=Ward(single_sex_bays=False), seed=SEED)
    a_ = s2.set_index("run").deaths_in_hdu_per_1000
    b_ = sim[sim.policy == "CP-SAT"].set_index("run").deaths_in_hdu_per_1000
    diff = a_ - b_
    summary["no_bay_constraint_cp_per_1000"] = float(a_.mean())
    summary["bay_rule_cost_per_1000"] = float(diff.mean())
    summary["bay_rule_cost_ci"] = [float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                                   float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))]
    summary["rrr_assumed"] = rrr
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary, ps


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--runs", type=int, default=200, help="simulation replications (90 days each)")
    ap.add_argument("--rrr", type=float, default=0.15, help="assumed relative risk reduction of HDU care")
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap samples for a fast check")
    a = ap.parse_args()
    summ, ps = run(a.data_dir, a.out_dir, a.runs, a.rrr, a.quick)
    print(json.dumps(summ, indent=2, default=float))
    print(ps.round(2).to_string(index=False))
