import pandas as pd
import numpy as np
import pyomo.environ as pyo
from collections import defaultdict
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import warnings

# Constants
WIND_CAP = 2000.0
THERMAL_CAP = 7000.0
NSP_CAP = 10000.0
VC_t, VC_w, VC_nsp = 24, 3, 5000
CN_f = 1
RU, RD = 1400.0, 1400.0
Lines = [("Bus1", "Bus2"), ("Bus1", "Bus3"), ("Bus2", "Bus3")]
LineLimits = {("Bus1", "Bus2"): 4000.0, ("Bus1", "Bus3"): 3000.0, ("Bus2", "Bus3"): 2000.0}
LineLimits.update({(j, i): v for (i, j), v in LineLimits.items()})
buses = ["Bus1", "Bus2", "Bus3"]


def load_and_prepare_data(path):
    """
    Load and preprocess time series data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file containing hourly time series data.

    Returns
    -------
    df : pandas.DataFrame
        Preprocessed DataFrame with columns:
        - Demand: actual load
        - Wind: onshore wind generation
        - Hour: integer index representing each hour
        - Wind_CF: wind capacity factor (Wind / WIND_CAP)
        - Demand_Bus1, Demand_Bus2, Demand_Bus3: nodal demand values
    hours : list of int
        List of hourly indices corresponding to rows in df.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.loc["2015-01-01":"2015-12-31", [
        "AT_load_actual_entsoe_transparency",
        "AT_wind_onshore_generation_actual"
    ]].copy()
    df.columns = ["Demand", "Wind"]
    df.interpolate(inplace=True)
    df = df.reset_index(drop=True)
    df["Hour"] = df.index
    df["Wind_CF"] = df["Wind"] / WIND_CAP
    df["Demand_Bus1"] = df["Demand"]
    df["Demand_Bus2"] = 0.0
    df["Demand_Bus3"] = 0.0
    return df, df["Hour"].tolist()


def build_full_model(df, hours):
    """
    Construct the full-scale unit commitment and dispatch model as a Pyomo ConcreteModel.

    Sets
    ----
    T : hours
    B : buses
    L : transmission line tuples

    Variables
    ---------
    P_t : thermal generation output at Bus1
    P_w : wind generation output at Bus2
    NSP : non-served power at all buses
    Theta : voltage angle at each bus
    F : power flow on each line
    F_abs : absolute value of power flow (for cost penalization)

    Objective
    ---------
    Minimize total variable production costs, unserved energy cost, and flow penalty.

    Constraints
    -----------
    - Nodal power balance
    - DC flow equations and line limits
    - Generation capacity limits and ramping
    - Slack bus reference angle

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing demand and wind CF by hour.
    hours : list of int
        Hours indices for model time set.

    Returns
    -------
    model : pyo.ConcreteModel
        Fully instantiated optimization model.
    """
    model = pyo.ConcreteModel("Full")
    model.T = pyo.Set(initialize=hours)
    model.B = pyo.Set(initialize=buses)
    model.L = pyo.Set(dimen=2, initialize=LineLimits.keys())

    model.P_t = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.P_w = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.NSP = pyo.Var(model.B, model.T, within=pyo.NonNegativeReals)
    model.Theta = pyo.Var(model.B, model.T)
    model.F = pyo.Var(model.L, model.T)
    model.F_abs = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)

    model.Obj = pyo.Objective(
        expr=sum(
            VC_t * model.P_t[t] + VC_w * model.P_w[t] +
            VC_nsp * sum(model.NSP[b, t] for b in model.B)
            for t in model.T
        ) + sum(CN_f * model.F_abs[l, t] for l in model.L for t in model.T),
        sense=pyo.minimize
    )

    def balance(m, b, t):
        """
        Nodal power balance at bus b and time t.
        """
        gen = 0.0
        if b == "Bus1": gen += m.P_t[t]
        if b == "Bus2": gen += m.P_w[t]
        inflow = sum(m.F[i, t] for i in m.L if i[1] == b)
        outflow = sum(m.F[i, t] for i in m.L if i[0] == b)
        demand = df.loc[t, f"Demand_{b}"]
        return gen + inflow - outflow + m.NSP[b, t] == demand

    model.balance = pyo.Constraint(model.B, model.T, rule=balance)
    model.flow_eq = pyo.Constraint(model.L, model.T,
        rule=lambda m, i, j, t: m.F[i, j, t] == m.Theta[i, t] - m.Theta[j, t]
    )
    model.flow_up = pyo.Constraint(model.L, model.T,
        rule=lambda m, i, j, t: m.F[i, j, t] <= LineLimits[i, j]
    )
    model.flow_lo = pyo.Constraint(model.L, model.T,
        rule=lambda m, i, j, t: -m.F[i, j, t] <= LineLimits[i, j]
    )
    model.abs_pos = pyo.Constraint(model.L, model.T,
        rule=lambda m, i, j, t: m.F_abs[i, j, t] >= m.F[i, j, t]
    )
    model.abs_neg = pyo.Constraint(model.L, model.T,
        rule=lambda m, i, j, t: m.F_abs[i, j, t] >= -m.F[i, j, t]
    )
    model.thermal_cap = pyo.Constraint(model.T,
        rule=lambda m, t: m.P_t[t] <= THERMAL_CAP
    )
    model.wind_cap = pyo.Constraint(model.T,
        rule=lambda m, t: m.P_w[t] <= df.loc[t, "Wind_CF"] * WIND_CAP
    )
    model.nsp_cap = pyo.Constraint(model.B, model.T,
        rule=lambda m, b, t: m.NSP[b, t] <= NSP_CAP
    )
    model.ramp_up = pyo.Constraint(model.T,
        rule=lambda m, t: pyo.Constraint.Skip if t == 0 else m.P_t[t] - m.P_t[t-1] <= RU
    )
    model.ramp_down = pyo.Constraint(model.T,
        rule=lambda m, t: pyo.Constraint.Skip if t == 0 else m.P_t[t-1] - m.P_t[t] <= RD
    )
    model.slack = pyo.Constraint(model.T,
        rule=lambda m, t: m.Theta["Bus1", t] == 0
    )
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    return model


def solve_model(model):
    """
    Solve a Pyomo model using the GLPK solver.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Instantiated Pyomo model to solve.
    """
    solver = pyo.SolverFactory("glpk")
    solver.solve(model)


def collect_full_outputs(model, df, hours):
    """
    Extract solution values from the full model and append to the DataFrame.

    Adds columns:
    - Thermal, Wind: generation outputs
    - NSP_<bus>: non-served power by bus
    - NSP_Total: total unserved energy
    - RU_dual, RD_dual, Wind_dual: dual values for ramp and wind constraints
    - MC_<bus>: market clearing price (dual of balance)

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved full model with dual suffix.
    df : pandas.DataFrame
        DataFrame updated in-place with solution values.
    hours : list of int
        Hour indices for iteration.
    """
    df["Thermal"] = [pyo.value(model.P_t[t]) for t in hours]
    df["Wind"] = [pyo.value(model.P_w[t]) for t in hours]
    for b in buses:
        df[f"NSP_{b}"] = [pyo.value(model.NSP[b, t]) for t in hours]
    df["NSP_Total"] = df[[f"NSP_{b}" for b in buses]].sum(axis=1)
    df["RU_dual"] = [0.0] + [model.dual.get(model.ramp_up[t], 0.0) for t in hours[1:]]
    df["RD_dual"] = [0.0] + [model.dual.get(model.ramp_down[t], 0.0) for t in hours[1:]]
    df["Wind_dual"] = [model.dual.get(model.wind_cap[t], 0.0) for t in hours]
    for b in buses:
        df[f"MC_{b}"] = [model.dual.get(model.balance[b, t], 0.0) for t in hours]


def process_basis(df, hours):
    """
    Identify contiguous time windows where the binding basis changes.

    Uses marginal cost and dual values to classify hours into basis-constant intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains columns MC_Bus1, RU_dual, RD_dual, Wind_dual for each hour.
    hours : list of int
        Hour indices.

    Returns
    -------
    subsets : list of lists
        Each sublist is a sequence of hour indices forming a basis interval.
    """
    tol = 1e-6
    MC_arr = df["MC_Bus1"].values
    DualRUP = df["RU_dual"].values
    DualRDN = df["RD_dual"].values
    DualWnd = df["Wind_dual"].values

    Basis = [False] * len(hours)
    Lengths = {}
    for mc in np.unique(np.round(MC_arr, 6)):
        if mc > tol and abs(mc - VC_t) > tol and abs(mc - VC_w) > tol:
            Lengths[mc] = max(1, int(round(mc / VC_t)))

    t_idx = 0
    while t_idx < len(hours):
        mc = MC_arr[t_idx]
        if abs(mc - VC_t) <= tol or abs(mc - VC_w) <= tol:
            t_idx += 1
            continue
        if mc < -tol:
            t2 = next((j for j in range(t_idx+1, len(hours))
                       if abs(MC_arr[j] - VC_nsp) <= tol), None)
            if t2 is not None:
                for k in range(t_idx, t2+1): Basis[k] = True
                t_idx = t2 + 1
                continue
        elif abs(mc - VC_nsp) <= tol:
            t2 = next((j for j in range(t_idx+1, len(hours))
                       if abs(MC_arr[j] - VC_t) <= tol or abs(MC_arr[j] - VC_w) <= tol), None)
            if t2 is not None:
                for k in range(t_idx, t2+1): Basis[k] = True
                t_idx = t2 + 1
                continue
        else:
            l = Lengths.get(round(mc, 6), 1)
            if t_idx > 0 and DualRUP[t_idx-1] > tol:
                search_from = max(0, t_idx - l)
                t2 = next((j for j in range(t_idx-1, search_from-1, -1)
                           if DualWnd[j] > tol), None)
                if t2 is not None:
                    for k in range(t2, t_idx+1): Basis[k] = True
                    t_idx = t2 + 1
                    continue
            if t_idx > 0 and DualRDN[t_idx-1] > tol:
                t2 = next((j for j in range(t_idx+1, min(len(hours), t_idx+l+1))
                           if DualWnd[j] > tol), None)
                if t2 is not None:
                    for k in range(t_idx, t2+1): Basis[k] = True
                    t_idx = t2 + 1
                    continue
        t_idx += 1

    subsets = []
    i = 0
    while i < len(hours):
        if Basis[i]:
            j = i
            while j < len(hours) and Basis[j]: j += 1
            subsets.append(list(range(i, j)))
            i = j
        else:
            subsets.append([i])
            i += 1
    return subsets


def group_and_centroid(df, subsets):
    """
    Group basis intervals by pattern and compute centroid time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains Demand_Bus1 and Wind_CF series.
    subsets : list of lists
        Basis intervals (hour index windows).

    Returns
    -------
    agg_df : pandas.DataFrame
        Each row corresponds to a position within a window-length group,
        with columns:
        - Length: window length
        - WindowPos: position index within the window
        - Centroid_Demand_Bus1, Centroid_Wind_CF: average profiles
        - Weight: number of windows in the group
    group_map : dict
        Mapping from (length, pattern) to list of windows having that pattern.
    """
    group_map = defaultdict(list)
    for window in subsets:
        pattern = tuple(
            (round(df.loc[hr, "MC_Bus1"], 6),
             round(df.loc[hr, "RU_dual"], 6),
             round(df.loc[hr, "RD_dual"], 6),
             round(df.loc[hr, "Wind_dual"], 6))
            for hr in window
        )
        group_map[(len(window), pattern)].append(window)

    rows = []
    for (length, pattern), windows in group_map.items():
        demand_arr = np.stack([df.loc[win, "Demand_Bus1"].values for win in windows])
        wind_arr = np.stack([df.loc[win, "Wind_CF"].values for win in windows])
        cd = demand_arr.mean(axis=0)
        cw = wind_arr.mean(axis=0)
        for pos in range(length):
            rows.append({
                "Length": length,
                "WindowPos": pos,
                "Centroid_Demand_Bus1": cd[pos],
                "Centroid_Wind_CF": cw[pos],
                "Weight": len(windows)
            })
    return pd.DataFrame(rows), group_map


def build_agg_model(agg_df):
    """
    Construct the aggregated dispatch model based on centroid representation.

    Parameters
    ----------
    agg_df : pandas.DataFrame
        Aggregated centroid time series with Weight, Centroid_Demand_Bus1, Centroid_Wind_CF.

    Returns
    -------
    m : pyo.ConcreteModel
        Pyomo model over aggregated time steps.
    """
    idx = agg_df.index.tolist()
    wt = dict(zip(idx, agg_df["Weight"]))
    dem = dict(zip(idx, agg_df["Centroid_Demand_Bus1"]))
    wf = dict(zip(idx, agg_df["Centroid_Wind_CF"]))

    m = pyo.ConcreteModel("Agg")
    m.T = pyo.Set(initialize=idx)
    m.B = pyo.Set(initialize=buses)
    m.L = pyo.Set(dimen=2, initialize=LineLimits.keys())
    m.P_t = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_w = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.NSP = pyo.Var(m.B, m.T, within=pyo.NonNegativeReals)
    m.Theta = pyo.Var(m.B, m.T)
    m.F = pyo.Var(m.L, m.T)
    m.F_abs = pyo.Var(m.L, m.T, within=pyo.NonNegativeReals)

    def obj(m):
        """
        Objective: weighted sum of generation, unserved power costs, and flow penalties.
        """
        return sum(wt[t]*(VC_t*m.P_t[t] + VC_w*m.P_w[t] + VC_nsp*sum(m.NSP[b,t] for b in m.B)) for t in m.T) + sum(CN_f*wt[t]*m.F_abs[l,t] for l in m.L for t in m.T)
    m.Obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    def bal(m, b, t):
        """
        Aggregated nodal power balance at bus b and time index t.
        """
        g = m.P_t[t] if b=="Bus1" else (m.P_w[t] if b=="Bus2" else 0)
        inflow = sum(m.F[i,t] for i in m.L if i[1]==b)
        outflow = sum(m.F[i,t] for i in m.L if i[0]==b)
        return g + inflow - outflow + m.NSP[b,t] == (dem[t] if b=="Bus1" else 0)
    m.balance = pyo.Constraint(m.B, m.T, rule=bal)
    m.flow_eq = pyo.Constraint(m.L, m.T, rule=lambda m,i,j,t: m.F[i,j,t]==m.Theta[i,t]-m.Theta[j,t])
    m.flow_up = pyo.Constraint(m.L, m.T, rule=lambda m,i,j,t: m.F[i,j,t]<=LineLimits[i,j])
    m.flow_lo = pyo.Constraint(m.L, m.T, rule=lambda m,i,j,t:-m.F[i,j,t]<=LineLimits[i,j])
    m.abs_p = pyo.Constraint(m.L, m.T, rule=lambda m,i,j,t: m.F_abs[i,j,t]>=m.F[i,j,t])
    m.abs_n = pyo.Constraint(m.L, m.T, rule=lambda m,i,j,t: m.F_abs[i,j,t]>=-m.F[i,j,t])
    m.thermal_cap = pyo.Constraint(m.T, rule=lambda m,t: m.P_t[t]<=THERMAL_CAP)
    m.wind_cap = pyo.Constraint(m.T, rule=lambda m,t: m.P_w[t]<=wf[t]*WIND_CAP)
    m.nsp_cap = pyo.Constraint(m.B, m.T, rule=lambda m,b,t: m.NSP[b,t]<=NSP_CAP)
    m.slack = pyo.Constraint(m.T, rule=lambda m,t: m.Theta["Bus1",t]==0)
    return m


def compare_and_report(full, agg, df, agg_df):
    """
    Compare full and aggregated model objectives and report errors and reduction.

    Prints:
    - Full model objective value
    - Aggregated model objective value
    - Relative error (%)
    - Hours reduced count and percentage

    Parameters
    ----------
    full : pyo.ConcreteModel
        Solved full model.
    agg : pyo.ConcreteModel
        Solved aggregated model.
    df : pandas.DataFrame
        Full dataset used.
    agg_df : pandas.DataFrame
        Aggregated centroid dataset.
    """
    fobj = pyo.value(full.Obj)
    aobj = pyo.value(agg.Obj)
    err = 100*abs(fobj-aobj)/fobj
    print(f"Full model objective:       {fobj:,.2f} €")
    print(f"Aggregated model objective: {aobj:,.2f} €")
    print(f"Relative error:             {err:.3f}%")
    print(f"Hours reduced: {len(df)} → {len(agg_df)} ({100*(1-len(agg_df)/len(df)):.2f}%)")


def construct_table6(df, subsets, group_map):
    """
    Create and print summary table (Table 6) of subsets by length.

    Calculates for each length:
    - Number of subsets
    - Number of unique dual bases
    - Average and total objective contributions in first step

    Parameters
    ----------
    df : pandas.DataFrame
        Full solution outputs.
    subsets : list of lists
        Basis time windows.
    group_map : dict
        Mapping of patterns to windows.
    """
    length_cnt = defaultdict(int)
    for s in subsets:
        length_cnt[len(s)] += 1
    dual_bases = defaultdict(set)
    for (length, pat) in group_map:
        dual_bases[length].add(pat)
    rows = []
    for length in sorted(length_cnt):
        costs = []
        for s in subsets:
            if len(s)==length:
                c = df.loc[s, "Thermal"].sum()*VC_t + df.loc[s, "Wind"].sum()*VC_w + df.loc[s, "NSP_Total"].sum()*VC_nsp
                costs.append(c)
        rows.append({
            "Length (hours)": length,
            "# Subsets (1st step)": length_cnt[length],
            "# Bases (2nd step)": len(dual_bases[length]),
            "Obj. fun. avg. (1st step)": int(round(np.mean(costs))),
            "Sum obj. fun. (1st step)": int(round(sum(costs)))
        })
    table = pd.DataFrame(rows).sort_values("Length (hours)")
    print(table.to_string(index=False))


def plot_results(df, group_map):
    """
    Plot parallel coordinates of inputs and dual values by basis group.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns Wind_CF, Demand_Bus1, RU_dual, RD_dual, Wind_dual.
    group_map : dict
        Mapping of basis patterns to windows for coloring.
    """
    warnings.filterwarnings("ignore", message="No artists with labels found to put in legend.*")
    basis_id = {}
    for bid, ((length, pat), windows) in enumerate(group_map.items()):
        for win in windows:
            for hr in win:
                basis_id[hr] = bid
    df['BasisID'] = df.index.map(basis_id)
    cols = ['Wind_CF','Demand_Bus1','RU_dual','RD_dual','Wind_dual']
    plot_df = df[cols+['BasisID']].dropna().copy()
    if plot_df.empty:
        print("No data to plot.")
        return
    plot_df['BasisID'] = plot_df['BasisID'].astype(str)
    plt.figure(figsize=(12,6))
    ax = parallel_coordinates(plot_df, class_column='BasisID', cols=cols,
                               color=plt.cm.tab10.colors, linewidth=2, alpha=0.7)
    if ax.get_legend(): ax.get_legend().remove()
    labels = ['Wind Capacity\nFactor','Demand\n(MW)','Ramp UP dual\n(€/MW)','Ramp Down dual\n(€/MW)','Wind dual\n(€/MW)']
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold', rotation=15)
    ax.set_xlabel('Metric', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=16, fontweight='bold')
    ax.set_title('Inputs & Duals by Basis', fontsize=18, fontweight='bold')
    for lbl in ax.get_yticklabels(): lbl.set_fontweight('bold'); lbl.set_fontsize(20)
    plt.tight_layout(); plt.subplots_adjust(right=0.98); plt.show()


def main():
    """
    Execute full workflow:
    1. Load and prepare data
    2. Build, solve, and extract full model outputs
    3. Identify basis windows and compute centroids
    4. Build, solve aggregated model
    5. Compare and report results
    6. Construct summary table and plot results
    """
    path = "data/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv"
    df, hours = load_and_prepare_data(path)
    full = build_full_model(df, hours)
    solve_model(full)
    collect_full_outputs(full, df, hours)

    subsets = process_basis(df, hours)
    agg_df, group_map = group_and_centroid(df, subsets)

    agg = build_agg_model(agg_df)
    solve_model(agg)

    compare_and_report(full, agg, df, agg_df)
    construct_table6(df, subsets, group_map)
    plot_results(df, group_map)


if __name__ == "__main__":
    main()
