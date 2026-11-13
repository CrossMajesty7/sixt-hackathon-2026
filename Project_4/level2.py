"""Level 2: Level 1 plus Transfers per Driver zwischen Stationen.

Neu gegenueber level1.py (Regeln R3, R4, R10):
  tr[s,t,k,d]  Autos der Kategorie k, die am Tag d von s nach t fahren (ganzzahlig)
  f[s,k,d] = station_fleet + kum.add - kum.remove + Ankuenfte(<=d) - Abfahrten(<=d)
  Ankunft am Tag d + lead_time, nur wenn <= letzter Tag
  Holding: auch fuer Autos im Transit (= abgefahren, noch nicht angekommen)
  Kosten:  tr * (fixed + cost_per_km * km) fuer Driver
Nur Paare, bei denen ein Transfer billiger als ein add ist, werden zugelassen (Rechenzeit).
"""
import sys
from pathlib import Path

import pandas as pd
import pulp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fleetkit import baseline_plan, load_instance, write_plan  # noqa: E402
from level1 import (  # noqa: E402
    SHARES, buckets, build_inputs, check_concave, evaluate as evaluate_l1, is_concave, price_curves,
)

MAX_TRANSFER_COST_FACTOR = 1.0  # Paar nur zulassen, wenn Fahrtkosten <= Faktor * addition_cost


def allowed_pairs(instance):
    drv = instance.transfer_costs.set_index("mode").loc["driver"]
    dist = instance.distances.copy()
    dist["cost"] = float(drv["fixed_cost"]) + float(drv["cost_per_km"]) * dist["distance_km"]
    add_cost = float(instance.categories["addition_cost_per_car"].max())
    dist = dist[dist["cost"] <= MAX_TRANSFER_COST_FACTOR * add_cost]
    return {(r.from_station_id, r.to_station_id): (int(r.lead_time_days), float(r.cost))
            for r in dist.itertuples(index=False)}


def solve_level2(instance, df, curves, time_limit=240):
    caps = instance.categories.set_index("category")
    days = sorted(df["date"].unique())
    didx = {d: i for i, d in enumerate(days)}
    pairs = allowed_pairs(instance)
    keys = list(df[["station_id", "category"]].drop_duplicates().itertuples(index=False, name=None))
    cats = sorted(df["category"].unique())
    print(f"Transfer-Paare zugelassen: {len(pairs)} von {len(instance.distances)}")

    m = pulp.LpProblem("level2", pulp.LpMaximize)
    add = {(s, k, d): m.add_variable(f"add_{s}_{k}_{i}", lowBound=0, cat="Integer")
           for s, k in keys for i, d in enumerate(days)}
    rem = {(s, k, d): m.add_variable(f"rem_{s}_{k}_{i}", lowBound=0, cat="Integer")
           for s, k in keys for i, d in enumerate(days)}
    tr = {(s, t, k, d): m.add_variable(f"tr_{s}_{t}_{k}_{i}", lowBound=0, cat="Integer")
          for (s, t) in pairs for k in cats for i, d in enumerate(days)}

    # Abfahrten und Ankuenfte je (Station, Kategorie, Tag-Index)
    dep = {}   # (s,k,i) -> Liste von Variablen, die am Tag i abfahren
    arr = {}   # (t,k,i) -> Liste von Variablen, die am Tag i ankommen
    transit_days = {}  # (k) -> Liste (var, Anzahl Tage im Transit innerhalb des Horizonts)
    for (s, t, k, d), v in tr.items():
        i = didx[d]
        lead, _ = pairs[(s, t)]
        dep.setdefault((s, k, i), []).append(v)
        if i + lead < len(days):
            arr.setdefault((t, k, i + lead), []).append(v)
        # Transit-Tage: von Abfahrtstag bis Tag vor Ankunft (R10: Auto im Transit kostet Holding)
        n_transit = min(lead, len(days) - i)
        transit_days.setdefault(k, []).append((v, n_transit))

    obj = []
    for (s, k), grp in df.groupby(["station_id", "category"]):
        grp = grp.sort_values("date")
        hold = float(caps.loc[k, "holding_cost_per_day"])
        cum = 0
        for i, r in enumerate(grp.itertuples(index=False)):
            cum = cum + add[s, k, r.date] - rem[s, k, r.date]
            cum = cum + pulp.lpSum(arr.get((s, k, i), [])) - pulp.lpSum(dep.get((s, k, i), []))
            f = r.station_fleet + cum
            m += f >= r.committed_car_days, f"valid_{s}_{k}_{i}"
            obj.append(-hold * f)

            if r.demand_to_come <= 1e-9:
                continue
            curve = curves[(r.station_type, k)]
            bs = buckets(curve)
            served = []
            for b, (dq, marg) in enumerate(bs):
                v = m.add_variable(f"srv_{s}_{k}_{i}_{b}", lowBound=0, upBound=dq * r.demand_to_come)
                served.append(v)
                obj.append(marg * r.base_revenue * v)
            if not is_concave(curve):
                for b in range(len(bs) - 1):
                    y = m.add_variable(f"y_{s}_{k}_{i}_{b}", cat="Binary")
                    m += served[b] >= bs[b][0] * r.demand_to_come * y
                    m += served[b + 1] <= bs[b + 1][0] * r.demand_to_come * y
            z = m.add_variable(f"z_{s}_{k}_{i}", cat="Binary")
            big = r.demand_to_come + r.committed_car_days
            m += pulp.lpSum(served) <= r.max_utilization * f - r.committed_car_days + big * (1 - z)
            m += pulp.lpSum(served) <= r.demand_to_come * z
            m += r.max_utilization * f - r.committed_car_days >= -r.committed_car_days * (1 - z)

    # Kosten der Transfers und Holding im Transit
    for (s, t, k, d), v in tr.items():
        obj.append(-pairs[(s, t)][1] * v)
    for k, lst in transit_days.items():
        hold = float(caps.loc[k, "holding_cost_per_day"])
        obj.append(-hold * pulp.lpSum(n * v for v, n in lst))

    for k in caps.index:
        adds = pulp.lpSum(v for (s, kk, d), v in add.items() if kk == k)
        rems = pulp.lpSum(v for (s, kk, d), v in rem.items() if kk == k)
        m += adds <= float(caps.loc[k, "addition_cap"])
        m += rems <= float(caps.loc[k, "removal_cap"])
        obj.append(-float(caps.loc[k, "addition_cost_per_car"]) * adds)

    m += pulp.lpSum(obj)
    stats = m.solve(pulp.COIN_CMD(msg=False, timeLimit=time_limit, gapRel=0.005))
    if not stats.has_solution:
        raise RuntimeError(f"Keine Loesung: {stats.status}")
    print("Status:", stats.status.name)

    rows = []
    for action, var in [("add", add), ("remove", rem)]:
        for (s, k, d), v in var.items():
            n = round(v.value() or 0)
            if n > 0:
                rows.append((action, d, k, s, None, None, n))
    for (s, t, k, d), v in tr.items():
        n = round(v.value() or 0)
        if n > 0:
            rows.append(("transfer", d, k, s, t, "driver", n))
    return pd.DataFrame(rows, columns=["action", "date", "category", "station_id",
                                        "to_station_id", "mode", "cars"])


def evaluate(plan, instance, df, curves):
    """Lokale Naeherung mit Transfers (ohne Upgrades). Ohne Umsatz der committed Mieten."""
    import numpy as np
    cats = instance.categories.set_index("category")
    dist = instance.distances.set_index(["from_station_id", "to_station_id"])
    drv = instance.transfer_costs.set_index("mode").loc["driver"]
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
    transfer_cost = 0.0
    transit_holding = 0.0
    for r in p[p["action"] == "transfer"].itertuples(index=False):
        lead = int(dist.loc[(r.station_id, r.to_station_id), "lead_time_days"])
        km = float(dist.loc[(r.station_id, r.to_station_id), "distance_km"])
        transfer_cost += r.cars * (float(drv["fixed_cost"]) + float(drv["cost_per_km"]) * km)
        d.loc[(d.station_id == r.station_id) & (d.category == r.category) & (d.date == r.date), "n_dep"] += r.cars
        arr_day = r.date + pd.Timedelta(days=lead)
        if arr_day <= last:
            d.loc[(d.station_id == r.to_station_id) & (d.category == r.category) & (d.date == arr_day), "n_arr"] += r.cars
        n_transit = min(lead, (last - r.date).days + 1)
        transit_holding += r.cars * n_transit * float(cats.loc[r.category, "holding_cost_per_day"])

    g = d.groupby(["station_id", "category"])
    d["fleet"] = (d["station_fleet"] + g["n_add"].cumsum() - g["n_rem"].cumsum()
                  + g["n_arr"].cumsum() - g["n_dep"].cumsum())

    valid = bool((d["fleet"] >= d["committed_car_days"]).all())
    tot = p.groupby(["action", "category"])["cars"].sum()
    for k in cats.index:
        valid &= bool(tot.get(("add", k), 0) <= cats.loc[k, "addition_cap"])
        valid &= bool(tot.get(("remove", k), 0) <= cats.loc[k, "removal_cap"])

    cap = (d["max_utilization"] * d["fleet"] - d["committed_car_days"]).clip(lower=0)
    dem = d["demand_to_come"]
    served = np.minimum(dem, cap)
    q = (served / dem.where(dem > 0, 1)).where(dem > 0, 0.0)
    revenue = 0.0
    for (stype, k), idx in d.groupby(["station_type", "category"]).groups.items():
        gq = np.interp(q.loc[idx], SHARES, curves[(stype, k)])
        revenue += float((gq * dem.loc[idx] * d.loc[idx, "base_revenue"]).sum())
    holding = float((d["fleet"] * d["category"].map(cats["holding_cost_per_day"])).sum()) + transit_holding
    addcost = float((d["n_add"] * d["category"].map(cats["addition_cost_per_car"])).sum())
    return {"gueltig": valid, "profit": revenue - holding - addcost - transfer_cost,
            "umsatz_neu": revenue, "holding": holding, "addition": addcost, "transfer": transfer_cost,
            "add": int(d["n_add"].sum()), "remove": int(d["n_rem"].sum()),
            "transfers": int(p.loc[p["action"] == "transfer", "cars"].sum())}


if __name__ == "__main__":
    instance = load_instance(data_dir=HERE / "data")
    df = build_inputs(instance)
    curves = price_curves(instance)
    check_concave(curves)

    plan = solve_level2(instance, df, curves)
    out = HERE / "plans" / "level2_v1.csv"
    write_plan(df_plan=plan, path=out)
    print(f"Plan: {len(plan)} Zeilen -> {out}")

    print("\nLokale Auswertung (Prognose als Szenario, ohne Upgrades, ohne committed Umsatz):")
    cols = ["action", "date", "category", "station_id", "to_station_id", "mode", "cars"]
    l1_path = HERE / "plans" / "level1_v1.csv"
    rows = [("do_nothing", pd.DataFrame(columns=cols)),
            ("baseline", baseline_plan(instance=instance))]
    if l1_path.exists():
        rows.append(("level1_v1", pd.read_csv(l1_path)))
    rows.append(("level2_v1", plan))
    for name, p in rows:
        r = evaluate(p, instance, df, curves)
        print(f"  {name:11s} gueltig={r['gueltig']}  profit={r['profit']:>10,.0f}  umsatz={r['umsatz_neu']:>10,.0f}  "
              f"transfer={r['transfer']:>7,.0f}  add={r['add']} remove={r['remove']} transfers={r['transfers']}")
