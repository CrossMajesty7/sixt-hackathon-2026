import pandas as pd
from pathlib import Path
from fleetkit import load_instance

HERE = Path(__file__).resolve().parent

def compute_demand_forecast(instance, level0_df):
    """
    Berechnet den Demand to Come (Neunachfrage) für den 14-Tage-Horizont
    basierend auf der 8-Wochen-Historie minus committed_car_days,
    sowie den Base Revenue pro Station und Kategorie.
    """
    # 0. Base Revenue pro Station und Kategorie berechnen
    completed_rev = instance.reservations.query("status == 'completed'").copy()
    completed_rev["pickup_date"] = pd.to_datetime(completed_rev["pickup_date"])
    completed_rev["return_date"] = pd.to_datetime(completed_rev["return_date"])
    
    # Mietdauer berechnen (return_date - pickup_date + 1), mindestens 1 Tag
    completed_rev["duration"] = (completed_rev["return_date"] - completed_rev["pickup_date"]).dt.days + 1
    completed_rev["duration"] = completed_rev["duration"].apply(lambda x: max(1, x))
    completed_rev["rev_per_day"] = completed_rev["revenue"] / completed_rev["duration"]

    base_revenue_df = (
        completed_rev.groupby(["pickup_station_id", "category"])["rev_per_day"]
        .mean()
        .reset_index(name="base_revenue")
    )

    # 1. Nur completed Mieten aus der Historie (letzte 8 Wochen) für Demand
    completed = instance.reservations.query("status == 'completed'").copy()
    completed["pickup_date"] = pd.to_datetime(completed["pickup_date"])
    completed["return_date"] = pd.to_datetime(completed["return_date"])

    # 2. Jede Reservierung über ihre Mietdauer auf die einzelnen Tage entfalten
    rows = []
    for _, r in completed.iterrows():
        for d in pd.date_range(r["pickup_date"], r["return_date"], freq="D"):
            rows.append({
                "station_id": r["pickup_station_id"],
                "category": r["category"],
                "date": d
            })
    
    hist_days = pd.DataFrame(rows)
    hist_days["weekday"] = hist_days["date"].dt.weekday  # 0 = Montag, 6 = Sonntag

    # 3. Zählen, wie viele Autos an welchem Kalendertag in der Historie vermietet waren
    daily_cars = (
        hist_days.groupby(["station_id", "category", "date"])
        .size()
        .reset_index(name="cars_on_rent")
    )
    daily_cars["weekday"] = daily_cars["date"].dt.weekday

    # 4. Durchschnittliche Anzahl vermieteter Autos pro Station, Kategorie und Wochentag über 8 Wochen
    weekly_mean = (
        daily_cars.groupby(["station_id", "category", "weekday"])["cars_on_rent"]
        .sum()
        .reset_index(name="total_car_days")
    )
    weekly_mean["mean_cars_on_rent"] = weekly_mean["total_car_days"] / 8.0

    # 5. Auf den 14-Tage-Horizont übertragen (aus eurem Level 0 Dataframe)
    forecast = level0_df[["station_id", "category", "date", "committed_car_days"]].copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["weekday"] = forecast["date"].dt.weekday

    # Historischen Mittelwert per Wochentag matchen
    forecast = forecast.merge(
        weekly_mean[["station_id", "category", "weekday", "mean_cars_on_rent"]],
        on=["station_id", "category", "weekday"],
        how="left"
    )
    forecast["mean_cars_on_rent"] = forecast["mean_cars_on_rent"].fillna(0)

    # 6. Demand to come = Historischer Schnitt minus committed_car_days (mindestens 0)
    forecast["demand_to_come"] = (
        forecast["mean_cars_on_rent"] - forecast["committed_car_days"]
    ).clip(lower=0)

    # 7. Base Revenue pro Station und Kategorie an den Forecast matchen
    forecast = forecast.merge(
        base_revenue_df,
        left_on=["station_id", "category"],
        right_on=["pickup_station_id", "category"],
        how="left"
    )
    if "pickup_station_id" in forecast.columns:
        forecast = forecast.drop(columns=["pickup_station_id"])
    forecast["base_revenue"] = forecast["base_revenue"].fillna(0)

    # Aufräumen und mit base_revenue zurückgeben
    return forecast[["station_id", "category", "date", "demand_to_come", "base_revenue"]]

if __name__ == "__main__":
    # Testen
    instance = load_instance(data_dir=HERE / "data")
    
    # Level-0-Daten einlesen
    level0_df = pd.read_csv(HERE / "level0_output.csv", parse_dates=["date"])
    
    df_demand = compute_demand_forecast(instance, level0_df)
    print("Beispiel für den Forecast mit Base Revenue:")
    print(df_demand.head(10))
    
    # Speichern
    df_demand.to_csv(HERE / "demand_forecast_output.csv", index=False)
    print("\nForecast erfolgreich gespeichert unter demand_forecast_output.csv")