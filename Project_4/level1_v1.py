"""Level 1: Flottengroesse planen (add / remove, keine Transfers, keine Upgrades).
Inklusive empirischer, gecappter Mietdauer-Verteilung (90%-Cutoff) aus den Reservierungen.

Modell (ein MIP ueber alle Kategorien, die Caps gelten je Kategorie):
  f[s,k,d] = station_fleet_do_nothing[s,k,d] + kum. add - kum. remove        (R3, R4)
  f >= committed                                                            (R5)
  served <= max(0, max_util * f - committed)                                (R6)
  Zielfunktion: Bucket-Umsatz - Holding * f - Addition * add                  (R8, R10)
Der Umsatz der committed Mieten ist vom Plan unabhaengig und fehlt daher im Ziel.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fleetkit import baseline_plan, load_instance, write_plan  # noqa: E402
from forecast import compute_demand_forecast  # noqa: E402
from level0 import compute_level0  # noqa: E402

SHARES = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]


def get_capped_duration_weights(instance):
    """Berechnet die empirische Mietdauer-Verteilung mit 90%-Cutoff und Capping.
    Alles über dem 90%-Perzentil wird auf den Cutoff-Tag gecappt (Rest gibt am letzten Tag ab).
    """
    df_res = instance.reservations.copy()
    df_res["pickup_date"] = pd.to_datetime(df_res["pickup_date"])
    df_res["return_date"] = pd.to_datetime(df_res["return_date"])
    df_res = df_res[df_res["status"] == "completed"].copy()
    
    # Exakte Mietdauer in Tagen
    df_res["duration"] = (df_res["return_date"] - df_res["pickup_date"]).dt.days + 1
    df_res["duration"] = df_res["duration"].apply(lambda x: max(1, x))
    
    detailed_rows = []
    grouped = df_res.groupby(["pickup_station_id", "category"])
    
    for (station, category), group in grouped:
        durations = group["duration"].copy()
        total_count = len(durations)
        if total_count == 0:
            continue
            
        # 90%-Cutoff bestimmen
        temp_counts = durations.value_counts().sort_index()
        temp_probs = temp_counts / total_count
        temp_cum_probs = temp_probs.cumsum()
        
        cutoff_day = None
        for duration, cum_p in temp_cum_probs.items():
            if cum_p >= 0.90:
                cutoff_day = duration
                break
        if cutoff_day is None:
            cutoff_day = durations.max()
            
        # Cappen: Alles > cutoff_day wird auf cutoff_day gesetzt
        durations = durations.apply(lambda d: min(d, cutoff_day))
        counts = durations.value_counts().sort_index()
        
        for duration, count in counts.items():
            share = count / total_count
            detailed_rows.append({
                "station_id": station,
                "category": category,
                "duration_days": duration,
                "share_percent": share * 100
            })
            
    return pd.DataFrame(detailed_rows)


def build_inputs(instance):
    """Eine Zeile pro Zelle (Station, Kategorie, Tag) inkl. empirischer Durchschnittsdauer."""
    l0 = compute_level0(instance)
    fc = compute_demand_forecast(instance, l0)  # demand_to_come + base_revenue

    # Empirische, gecappte Mietdauer-Verteilung laden und gewichtete Durchschnittsdauer berechnen
    dur_df = get_capped_duration_weights(instance)
    if not dur_df.empty:
        dur_df["weighted_days"] = dur_df["duration_days"] * (dur_df["share_percent"] / 100.0)
        avg_dur = dur_df.groupby(["station_id", "category"])["weighted_days"].sum().reset_index(name="avg_duration")
    else:
        avg_dur = pd.DataFrame(columns=["station_id", "category", "avg_duration"])

    df = l0.merge(fc, on=["station_id", "category", "date"], how="left")
    df = df.merge(instance.stations[["station_id", "station_type", "max_utilization"]], on="station_id", how="left")
    
    if not avg_dur.empty:
        df = df.merge(avg_dur, on=["station_id", "category"], how="left")
        df["avg_duration"] = df["avg_duration"].fillna(1.0)
    else:
        df["avg_duration"] = 1.0

    return df.sort_values(["station_id", "category", "date"]).reset_index(drop=True)


def price_curves(instance):
    """g(q) = q * price_factor(q) an den Stuetzstellen SHARES, je (Stationstyp, Kategorie)."""
    curves = {}
    for (stype, cat), grp in instance.price_factors.groupby(["station_type", "category"]):
        pf = {round(q, 1): f for q, f in zip(grp["served_share"], grp["price_factor"])}
        pf[0.0] = pf[0.6]  # unter 60 % gilt der Faktor von 60 %
        curves[(stype, cat)] = np.array([q * pf[q] for q in SHARES])
    return curves


def buckets(curve):
    """[(Groesse als Anteil der Nachfrage, Grenzerloes je Einheit Basisumsatz)] je Bucket."""
    out = []
    for i in range(1, len(SHARES)):
        dq = SHARES[i] - SHARES[i - 1]
        out.append((dq, (curve[i] - curve[i - 1]) / dq))
    return out


def is_concave(curve):
    marg = [m for _, m in buckets(curve)]
    return all(marg[i] >= marg[i + 1] - 1e-9 for i in range(len(marg) - 1))


def check_concave(curves):
    for key, curve in curves.items():
        if not is_concave(curve):
            marg = [m for _, m in buckets(curve)]
            print(f"WARNUNG: Grenzerloes steigt bei {key}: {np.round(marg, 3)} "
                  "-> Bucket-Reihenfolge wird per Binaervariablen erzwungen")


def solve_level1(instance, df, curves, time_limit=120):
    caps = instance.categories.set_index("category")
    days = sorted(df["date"].unique())
    keys = list(df[["station_id", "category"]].drop_duplicates().itertuples(index=False, name=None))

    m = pulp.LpProblem("level1", pulp.LpMaximize)
    add = {(s, k, d): m.add_variable(f"add_{s}_{k}_{i}", lowBound=0, cat="Integer")
           for s, k in keys for i, d in enumerate(days)}
    rem = {(s, k, d): m.add_variable(f"rem_{s}_{k}_{i}", lowBound=0, cat="Integer")
           for s, k in keys for i, d in enumerate(days)}

    obj = []
    for (s, k), grp in df.groupby(["station_id", "category"]):
        grp = grp.sort_values("date")
        hold = float(caps.loc[k, "holding_cost_per_day"])
        cum = 0
        for i, r in enumerate(grp.itertuples(index=False)):
            cum = cum + add[s, k, r.date] - rem[s, k, r.date]
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
                # Bucket b+1 darf erst gefuellt werden, wenn Bucket b voll ist
                for b in range(len(bs) - 1):
                    y = m.add_variable(f"y_{s}_{k}_{i}_{b}", cat="Binary")
                    m += served[b] >= bs[b][0] * r.demand_to_come * y
                    m += served[b + 1] <= bs[b + 1][0] * r.demand_to_come * y
            # R6: Kapazitaet a = max(0, util*f - committed). z=1: Kapazitaet >= 0, z=0: nichts bedient
            z = m.add_variable(f"z_{s}_{k}_{i}", cat="Binary")
            big = r.demand_to_come + r.committed_car_days
            m += pulp.lpSum(served) <= r.max_utilization * f - r.committed_car_days + big * (1 - z)
            m += pulp.lpSum(served) <= r.demand_to_come * z
            m += r.max_utilization * f - r.committed_car_days >= -r.committed_car_days * (1 - z)

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
                rows.append((action, d, k, s, n))
    plan = pd.DataFrame(rows, columns=["action", "date", "category", "station_id", "cars"])
    plan["to_station_id"] = None
    plan["mode"] = None
    return plan[["action", "date", "category", "station_id", "to_station_id", "mode", "cars"]]


def evaluate(plan, instance, df, curves):
    """Lokale Naeherung des Scorers (ohne Upgrades, ohne Transfers), Nachfrage = Prognose."""
    cats = instance.categories.set_index("category")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    p = plan.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["cars"] = p["cars"].astype(float)
    for action, col in [("add", "n_add"), ("remove", "n_rem")]:
        sub = p[p["action"] == action].groupby(["station_id", "category", "date"])["cars"].sum().rename(col)
        d = d.merge(sub, on=["station_id", "category", "date"], how="left")
        d[col] = d[col].fillna(0)
    d = d.sort_values(["station_id", "category", "date"]).reset_index(drop=True)
    g = d.groupby(["station_id", "category"])
    d["fleet"] = d["station_fleet"] + g["n_add"].cumsum() - g["n_rem"].cumsum()

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
    holding = float((d["fleet"] * d["category"].map(cats["holding_cost_per_day"])).sum())
    addcost = float((d["n_add"] * d["category"].map(cats["addition_cost_per_car"])).sum())
    return {"gueltig": valid, "profit": revenue - holding - addcost,
            "umsatz_neu": revenue, "holding": holding, "addition": addcost,
            "add": int(d["n_add"].sum()), "remove": int(d["n_rem"].sum())}


if __name__ == "__main__":
    instance = load_instance(data_dir=HERE / "data")
    df = build_inputs(instance)
    curves = price_curves(instance)
    check_concave(curves)

    plan = solve_level1(instance, df, curves)
    out = HERE / "plans" / "level1_v1.csv"
    write_plan(df_plan=plan, path=out)
    print(f"Plan: {len(plan)} Zeilen -> {out}")

    print("\nLokale Auswertung (Prognose als Szenario, ohne Upgrades, ohne committed Umsatz):")
    do_nothing = pd.DataFrame(columns=plan.columns)
    for name, p in [("do_nothing", do_nothing), ("baseline", baseline_plan(instance=instance)), ("level1_v1", plan)]:
        r = evaluate(p, instance, df, curves)
        print(f"  {name:11s} gueltig={r['gueltig']}  profit={r['profit']:>12,.0f}  "
              f"umsatz={r['umsatz_neu']:>11,.0f}  add={r['add']} remove={r['remove']}")