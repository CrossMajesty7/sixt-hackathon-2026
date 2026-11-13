"""Level 0: station fleet, committed car-days, one-way flows per station/category/day.

Compares computed values against data/check_values.csv.
"""
from pathlib import Path
import pandas as pd
from fleetkit import load_instance

HERE = Path(__file__).resolve().parent


def compute_level0(instance):
    horizon = instance.horizon.iloc[0]
    first_date = horizon["first_date"]
    last_date = horizon["last_date"]
    days = pd.date_range(first_date, last_date, freq="D")

    stations = instance.stations["station_id"].tolist()
    categories = instance.categories["category"].tolist()

    # Full grid of station x category x day
    grid = pd.DataFrame(
        [(s, k, d) for s in stations for k in categories for d in days],
        columns=["station_id", "category", "date"],
    )

    # ---- Initial fleet f[s,k,0]: idle snapshot cars + on-rent cars picked up at s ----
    idle = instance.fleet_snapshot.query("status == 'idle'")[["station_id", "category"]]
    on_rent = (
        instance.fleet_snapshot.query("status == 'on_rent'")
        .merge(instance.reservations[["reservation_id", "pickup_station_id"]], on="reservation_id")
        [["pickup_station_id", "category"]]
        .rename(columns={"pickup_station_id": "station_id"})
    )

    initial_fleet = (
        pd.concat([idle, on_rent], ignore_index=True)
        .groupby(["station_id", "category"])
        .size()
        .reset_index(name="fleet_day1")
    )

    # ---- Committed car-days: open + booked rentals, clipped to horizon ----
    committed = instance.reservations.query("status in ['open', 'booked']").copy()
    committed["start"] = committed["pickup_date"].clip(lower=first_date)
    committed["end"] = committed["return_date"].clip(upper=last_date)
    committed = committed[committed["start"] <= committed["end"]]

    rows = []
    for _, r in committed.iterrows():
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append(
                {
                    "station_id": r["pickup_station_id"],
                    "category": r["category"],
                    "date": d,
                }
            )
    committed_df = pd.DataFrame(rows)
    committed_counts = (
        committed_df.groupby(["station_id", "category", "date"])
        .size()
        .reset_index(name="committed_car_days")
    )

    # ---- One-way flows: open/booked rentals with return != pickup station ----
    oneway = committed[committed["return_station_id"] != committed["pickup_station_id"]].copy()
    oneway["flow_date"] = oneway["return_date"] + pd.Timedelta(days=1)
    oneway = oneway[oneway["flow_date"] <= last_date]

    departures = (
        oneway.groupby(["pickup_station_id", "category", "flow_date"])
        .size()
        .reset_index(name="oneway_departures")
        .rename(columns={"pickup_station_id": "station_id", "flow_date": "date"})
    )

    arrivals = (
        oneway.groupby(["return_station_id", "category", "flow_date"])
        .size()
        .reset_index(name="oneway_arrivals")
        .rename(columns={"return_station_id": "station_id", "flow_date": "date"})
    )

    # ---- Merge everything ----
    df = grid.merge(committed_counts, on=["station_id", "category", "date"], how="left")
    df = df.merge(departures, on=["station_id", "category", "date"], how="left")
    df = df.merge(arrivals, on=["station_id", "category", "date"], how="left")
    df = df.merge(initial_fleet, on=["station_id", "category"], how="left")

    for c in ["committed_car_days", "oneway_arrivals", "oneway_departures"]:
        df[c] = df[c].fillna(0).astype(int)
    df["fleet_day1"] = df["fleet_day1"].fillna(0).astype(int)

    df = df.sort_values(["station_id", "category", "date"])
    df["station_fleet"] = (
        df["fleet_day1"]
        + df.groupby(["station_id", "category"])["oneway_arrivals"].transform("cumsum")
        - df.groupby(["station_id", "category"])["oneway_departures"].transform("cumsum")
    )

    return df[
        [
            "station_id",
            "category",
            "date",
            "committed_car_days",
            "oneway_arrivals",
            "oneway_departures",
            "station_fleet",
        ]
    ]


if __name__ == "__main__":
    instance = load_instance(data_dir=HERE / "data")
    df = compute_level0(instance)

    check = pd.read_csv(HERE / "data" / "check_values.csv", parse_dates=["date"])
    merged = df.merge(check, on=["station_id", "category", "date"], suffixes=("_computed", "_check"))

    print(f"Computed {len(df)} cells; check values: {len(check)} cells; overlapping: {len(merged)}")
    all_ok = True
    for col in ["committed_car_days", "oneway_arrivals", "oneway_departures", "station_fleet"]:
        ok = (merged[f"{col}_computed"] == merged[f"{col}_check"]).all()
        print(f"  {col}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            all_ok = False
            mm = merged[merged[f"{col}_computed"] != merged[f"{col}_check"]]
            print(mm[["station_id", "category", "date", f"{col}_computed", f"{col}_check"]].to_string(index=False))

    if all_ok:
        print("\nAll check values match! Level 0 data pipeline is correct.")
    else:
        print("\nMISMATCHES FOUND - review the logic above.")

    out = HERE / "level0_output.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"\nFull output written to {out}")
