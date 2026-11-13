"""Level 3: Level 2 plus Upgrades zwischen Kategorien (R7) und Trucks mit 8 Plaetzen (R10).

Upgrades (je Station s, Tag d, Kategorien k=1..4 nach rank):
  own[k]   Nachfrage der Kategorie k, bedient mit Autos der Kategorie k
  up[k]    Nachfrage der Kategorie k-1, bedient mit Autos der Kategorie k (Umsatz der Kategorie k-1)
  served[k] = own[k] + up[k+1]  -> Buckets/Preiskurve der Kategorie k
  own[k] + up[k] <= max(0, util * f[k] - committed[k])
Trucks:
  trk[s,t,k,d] Autos per Truck, nt[s,t,d] Anzahl Trucks, sum_k trk <= 8 * nt, Kosten je Truck.
Driver wie in Level 2 (nur Paare, bei denen ein Driver billiger als ein add ist).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fleetkit import baseline_plan, load_instance, write_plan  # noqa: E402
from level1 import SHARES, buckets, build_inputs, check_concave, is_concave, price_curves  # noqa: E402
from level2 import allowed_pairs  # noqa: E402

TRUCK_CAP = 8


def truck_pairs(instance):
    t = instance.transfer_costs.set_index("mode").loc["truck"]
    dist = instance.distances
    return {(r.from_station_id, r.to_station_id): (int(r.lead_time_days),
            float(t["fixed_cost"]) + float(t["cost_per_km"]) * float(r.distance_km))
            for r in dist.itertuples(index=False)}


def solve_level3(instance, df, curves, time_limit=420, warm_plan=None):
    caps = instance.categories.set_index("category")
    cats = instance.categories.sort_values("rank")["category"].tolist()
    days = sorted(df["date"].unique())
    didx = {d: i for i, d in enumerate(days)}
    n_days = len(days)
    drv_pairs = allowed_pairs(instance)
    trk_pairs = truck_pairs(instance)
    stations = sorted(df["station_id"].unique())
    rec = {(r.station_id, r.category, didx[r.date]): r for r in df.itertuples(index=False)}
    print(f"Driver-Paare: {len(drv_pairs)}, Truck-Paare: {len(trk_pairs)}")

    m = pulp.LpProblem("level3", pulp.LpMaximize)
    add = {(s, k, d): m.add_variable(f"add_{s}_{k}_{i}", lowBound=0, cat="Integer")
           for s in stations for k in cats for i, d in enumerate(days)}
    rem = {(s, k, d): m.add_variable(f"rem_{s}_{k}_{i}", lowBound=0, cat="Integer")
           for s in stations for k in cats for i, d in enumerate(days)}
    trd = {(s, t, k, d): m.add_variable(f"trd_{s}_{t}_{k}_{i}", lowBound=0, cat="Integer")
           for (s, t) in drv_pairs for k in cats for i, d in enumerate(days)}
    trk = {(s, t, k, d): m.add_variable(f"trk_{s}_{t}_{k}_{i}", lowBound=0, cat="Integer")
           for (s, t) in trk_pairs for k in cats for i, d in enumerate(days)}
    nt = {(s, t, d): m.add_variable(f"nt_{s}_{t}_{i}", lowBound=0, cat="Integer")
          for (s, t) in trk_pairs for i, d in enumerate(days)}

    for (s, t) in trk_pairs:
        for d in days:
            m += pulp.lpSum(trk[s, t, k, d] for k in cats) <= TRUCK_CAP * nt[s, t, d]

    dep, arr, transit = {}, {}, {}
    for pairs, var in ((drv_pairs, trd), (trk_pairs, trk)):
        for (s, t, k, d), v in var.items():
            i = didx[d]
            lead = pairs[(s, t)][0]
            dep.setdefault((s, k, i), []).append(v)
            if i + lead < n_days:
                arr.setdefault((t, k, i + lead), []).append(v)
            transit.setdefault(k, []).append((v, min(lead, n_days - i)))

    obj = []
    fleet = {}
    for s in stations:
        for k in cats:
            hold = float(caps.loc[k, "holding_cost_per_day"])
            cum = 0
            for i in range(n_days):
                r = rec[(s, k, i)]
                cum = cum + add[s, k, days[i]] - rem[s, k, days[i]]
                cum = cum + pulp.lpSum(arr.get((s, k, i), [])) - pulp.lpSum(dep.get((s, k, i), []))
                f = r.station_fleet + cum
                fleet[(s, k, i)] = f
                m += f >= r.committed_car_days
                obj.append(-hold * f)

    for s in stations:
        for i in range(n_days):
            D = {k: rec[(s, k, i)].demand_to_come for k in cats}
            D = {k: (v if v > 1e-9 else 0.0) for k, v in D.items()}
            own, up, S = {}, {}, {}
            for j, k in enumerate(cats):
                r = rec[(s, k, i)]
                if D[k] > 0:
                    curve = curves[(r.station_type, k)]
                    bs = buckets(curve)
                    sv = []
                    for b, (dq, marg) in enumerate(bs):
                        v = m.add_variable(f"srv_{s}_{k}_{i}_{b}", lowBound=0, upBound=dq * D[k])
                        sv.append(v)
                        obj.append(marg * r.base_revenue * v)
                    if not is_concave(curve):
                        for b in range(len(bs) - 1):
                            y = m.add_variable(f"y_{s}_{k}_{i}_{b}", cat="Binary")
                            m += sv[b] >= bs[b][0] * D[k] * y
                            m += sv[b + 1] <= bs[b + 1][0] * D[k] * y
                    S[k] = pulp.lpSum(sv)
                    own[k] = m.add_variable(f"own_{s}_{k}_{i}", lowBound=0, upBound=D[k])
                if j >= 1 and D[cats[j - 1]] > 0:
                    up[k] = m.add_variable(f"up_{s}_{k}_{i}", lowBound=0, upBound=D[cats[j - 1]])
            for j, k in enumerate(cats):
                if D[k] > 0:
                    nxt = cats[j + 1] if j + 1 < len(cats) else None
                    m += S[k] == own[k] + (up[nxt] if nxt in up else 0)
            for j, k in enumerate(cats):
                used_terms = [v for v in (own.get(k), up.get(k)) if v is not None]
                if not used_terms:
                    continue
                r = rec[(s, k, i)]
                f = fleet[(s, k, i)]
                used = pulp.lpSum(used_terms)
                umax = D[k] + (D[cats[j - 1]] if j >= 1 else 0.0)
                z = m.add_variable(f"z_{s}_{k}_{i}", cat="Binary")
                big = umax + r.committed_car_days
                m += used <= r.max_utilization * f - r.committed_car_days + big * (1 - z)
                m += used <= umax * z
                m += r.max_utilization * f - r.committed_car_days >= -r.committed_car_days * (1 - z)

    for (s, t, d), v in nt.items():
        obj.append(-trk_pairs[(s, t)][1] * v)
    for (s, t, k, d), v in trd.items():
        obj.append(-drv_pairs[(s, t)][1] * v)
    for k, lst in transit.items():
        hold = float(caps.loc[k, "holding_cost_per_day"])
        obj.append(-hold * pulp.lpSum(n * v for v, n in lst))

    for k in cats:
        adds = pulp.lpSum(v for (s, kk, d), v in add.items() if kk == k)
        rems = pulp.lpSum(v for (s, kk, d), v in rem.items() if kk == k)
        m += adds <= float(caps.loc[k, "addition_cap"])
        m += rems <= float(caps.loc[k, "removal_cap"])
        obj.append(-float(caps.loc[k, "addition_cost_per_car"]) * adds)

    m += pulp.lpSum(obj)

    warm = False
    if warm_plan is not None and len(warm_plan):
        try:
            for v in list(add.values()) + list(rem.values()) + list(trd.values()) + list(trk.values()) + list(nt.values()):
                v.setInitialValue(0)
            wp = warm_plan.copy()
            wp["date"] = pd.to_datetime(wp["date"])
            for r in wp.itertuples(index=False):
                if r.action == "add":
                    add[r.station_id, r.category, r.date].setInitialValue(r.cars)
                elif r.action == "remove":
                    rem[r.station_id, r.category, r.date].setInitialValue(r.cars)
                elif r.action == "transfer" and r.mode == "driver":
                    trd[r.station_id, r.to_station_id, r.category, r.date].setInitialValue(r.cars)
            warm = True
        except Exception as e:  # Warmstart ist optional
            print("Warmstart nicht moeglich:", e)
    stats = m.solve(pulp.COIN_CMD(msg=True, logPath=str(HERE / "cbc_level3.log"), timeLimit=time_limit, gapRel=0.005, warmStart=warm))
    if not stats.has_solution:
        raise RuntimeError(f"Keine Loesung: {stats.status}")
    print("Status:", stats.status.name, "| Warmstart:", warm)

    rows = []
    for action, var in [("add", add), ("remove", rem)]:
        for (s, k, d), v in var.items():
            n = round(v.value() or 0)
            if n > 0:
                rows.append((action, d, k, s, None, None, n))
    for mode, var in (("driver", trd), ("truck", trk)):
        for (s, t, k, d), v in var.items():
            n = round(v.value() or 0)
            if n > 0:
                rows.append(("transfer", d, k, s, t, mode, n))
    return pd.DataFrame(rows, columns=["action", "date", "category", "station_id",
                                        "to_station_id", "mode", "cars"])


def evaluate(plan, instance, df, curves):
    """Lokale Naeherung nach R3-R10 mit exakter Greedy-Regel R7. Ohne Umsatz der committed Mieten."""
    cats_df = instance.categories.sort_values("rank").set_index("category")
    cats = cats_df.index.tolist()
    dist = instance.distances.set_index(["from_station_id", "to_station_id"])
    tc = instance.transfer_costs.set_index("mode")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    p = plan.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["cars"] = p["cars"].astype(float)
    last = d["date"].max()
    d = d.sort_values(["station_id", "category", "date"]).reset_index(drop=True)
    for action, col in [("add", "n_add"), ("remove", "n_rem")]:
        sub = p[p["action"] == action].groupby(["station_id", "category", "date"])["cars"].sum().rename(col)
        d = d.merge(sub, on=["station_id", "category", "date"], how="left")
        d[col] = d[col].fillna(0.0)
    d["n_dep"] = 0.0
    d["n_arr"] = 0.0
    transfer_cost, transit_hold = 0.0, 0.0
    tp = p[p["action"] == "transfer"]
    for r in tp.itertuples(index=False):
        lead = int(dist.loc[(r.station_id, r.to_station_id), "lead_time_days"])
        d.loc[(d.station_id == r.station_id) & (d.category == r.category) & (d.date == r.date), "n_dep"] += r.cars
        arr_day = r.date + pd.Timedelta(days=lead)
        if arr_day <= last:
            d.loc[(d.station_id == r.to_station_id) & (d.category == r.category) & (d.date == arr_day), "n_arr"] += r.cars
        transit_hold += r.cars * min(lead, (last - r.date).days + 1) * float(cats_df.loc[r.category, "holding_cost_per_day"])
    for r in tp[tp["mode"] == "driver"].itertuples(index=False):
        km = float(dist.loc[(r.station_id, r.to_station_id), "distance_km"])
        transfer_cost += r.cars * (float(tc.loc["driver", "fixed_cost"]) + float(tc.loc["driver", "cost_per_km"]) * km)
    for (s, t, _), grp in tp[tp["mode"] == "truck"].groupby(["station_id", "to_station_id", "date"]):
        km = float(dist.loc[(s, t), "distance_km"])
        n_trucks = int(np.ceil(grp["cars"].sum() / TRUCK_CAP - 1e-9))
        transfer_cost += n_trucks * (float(tc.loc["truck", "fixed_cost"]) + float(tc.loc["truck", "cost_per_km"]) * km)

    g = d.groupby(["station_id", "category"])
    d["fleet"] = (d["station_fleet"] + g["n_add"].cumsum() - g["n_rem"].cumsum()
                  + g["n_arr"].cumsum() - g["n_dep"].cumsum())
    valid = bool((d["fleet"] >= d["committed_car_days"]).all())
    tot = p.groupby(["action", "category"])["cars"].sum()
    for k in cats:
        valid &= bool(tot.get(("add", k), 0) <= cats_df.loc[k, "addition_cap"])
        valid &= bool(tot.get(("remove", k), 0) <= cats_df.loc[k, "removal_cap"])

    d["a"] = (d["max_utilization"] * d["fleet"] - d["committed_car_days"]).clip(lower=0)
    key = ["station_id", "date"]
    D = d.pivot(index=key, columns="category", values="demand_to_come")[cats].to_numpy()
    A = d.pivot(index=key, columns="category", values="a")[cats].to_numpy()
    own = np.minimum(D, A)
    up = np.zeros_like(D)  # up[:, j] = Autos der Kategorie j fuer Nachfrage der Kategorie j-1
    for j in range(1, len(cats)):
        up[:, j] = np.minimum(A[:, j] - own[:, j], D[:, j - 1] - own[:, j - 1])
    served = own.copy()
    served[:, :-1] += up[:, 1:]
    base = d.pivot(index=key, columns="category", values="base_revenue")[cats].to_numpy()
    stype = d.drop_duplicates("station_id").set_index("station_id")["station_type"]
    idx_st = d.pivot(index=key, columns="category", values="demand_to_come").index.get_level_values(0)
    st_arr = stype.reindex(idx_st).to_numpy()
    revenue = 0.0
    for j, k in enumerate(cats):
        q = np.where(D[:, j] > 0, served[:, j] / np.where(D[:, j] > 0, D[:, j], 1), 0.0)
        for ty in np.unique(st_arr):
            mk = st_arr == ty
            revenue += float((np.interp(q[mk], SHARES, curves[(ty, k)]) * D[mk, j] * base[mk, j]).sum())

    holding = float((d["fleet"] * d["category"].map(cats_df["holding_cost_per_day"])).sum()) + transit_hold
    addcost = float((d["n_add"] * d["category"].map(cats_df["addition_cost_per_car"])).sum())
    return {"gueltig": valid, "profit": revenue - holding - addcost - transfer_cost,
            "umsatz_neu": revenue, "transfer": transfer_cost,
            "upgrades": float(up[:, 1:].sum()),
            "add": int(d["n_add"].sum()), "remove": int(d["n_rem"].sum()),
            "drv": int(tp.loc[tp["mode"] == "driver", "cars"].sum()),
            "trk": int(tp.loc[tp["mode"] == "truck", "cars"].sum())}


if __name__ == "__main__":
    instance = load_instance(data_dir=HERE / "data")
    df = build_inputs(instance)
    curves = price_curves(instance)
    check_concave(curves)

    cols = ["action", "date", "category", "station_id", "to_station_id", "mode", "cars"]
    l2_path = HERE / "plans" / "level2_v1.csv"
    l2 = pd.read_csv(l2_path) if l2_path.exists() else None

    plan = solve_level3(instance, df, curves, warm_plan=l2)
    out = HERE / "plans" / "level3_v1.csv"
    write_plan(df_plan=plan, path=out)
    print(f"Plan: {len(plan)} Zeilen -> {out}")

    print("\nLokale Auswertung (Prognose als Szenario, ohne committed Umsatz):")
    rows = [("do_nothing", pd.DataFrame(columns=cols)), ("baseline", baseline_plan(instance=instance))]
    for name in ("level1_v1", "level2_v1"):
        pth = HERE / "plans" / f"{name}.csv"
        if pth.exists():
            rows.append((name, pd.read_csv(pth)))
    rows.append(("level3_v1", plan))
    for name, p in rows:
        r = evaluate(p, instance, df, curves)
        print(f"  {name:11s} gueltig={r['gueltig']}  profit={r['profit']:>10,.0f}  umsatz={r['umsatz_neu']:>10,.0f}  "
              f"transfer={r['transfer']:>7,.0f}  add={r['add']} rem={r['remove']} drv={r['drv']} trk={r['trk']}  "
              f"upgrade-cd={r['upgrades']:.0f}")
