"""Berechnet die exakte Verteilung der Mietdauern inklusive Einzelanteilen 
und kumulierten Wahrscheinlichkeiten (cum_probs) aus der reservations.csv.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

def analyze_detailed_durations():
    csv_path = HERE / "data" / "reservations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")
        
    df = pd.read_csv(csv_path, parse_dates=["pickup_date", "return_date"])
    
    # Nur abgeschlossene Buchungen betrachten
    df = df[df["status"] == "completed"].copy()
    
    # 1. Exakte Mietdauer in Tagen berechnen (inkl. Start- und Endtag)
    df["duration"] = (df["return_date"] - df["pickup_date"]).dt.days + 1
    df["duration"] = df["duration"].apply(lambda x: max(1, x)) # Mindestens 1 Tag
    
    detailed_rows = []
    
    # 2. Pro Station und Kategorie auswerten
    grouped = df.groupby(["pickup_station_id", "category"])
    
    for (station, category), group in grouped:
        durations = group["duration"]
        total_count = len(durations)
        
        if total_count == 0:
            continue
            
        # Häufigkeiten pro Dauer sortiert nach Tagen
        counts = durations.value_counts().sort_index()
        
        for duration, count in counts.items():
            share = count / total_count
            detailed_rows.append({
                "station_id": station,
                "category": category,
                "duration_days": duration,
                "count": count,
                "share_percent": round(share * 100, 2),       # Einzelner Anteil in %
                "cum_prob_percent": 0.0                        # Platzhalter für kumuliert
            })
            
    res_df = pd.DataFrame(detailed_rows)
    
    # 3. Kumulierte Wahrscheinlichkeit (cum_probs) pro Station und Kategorie berechnen
    res_df["cum_prob_percent"] = (
        res_df.groupby(["station_id", "category"])["share_percent"]
        .cumsum()
    )
    
    # 4. Speichern und Ausgeben
    out_path = HERE / "duration_distribution_detailed.csv"
    res_df.to_csv(out_path, index=False)
    
    print("--- Auszug der detaillierten Mietdauer-Verteilung ---")
    print(res_df.head(15).to_string(index=False))
    print(f"\nVollständige Tabelle mit allen Anteilen & cum_probs gespeichert unter: {out_path}")

if __name__ == "__main__":
    analyze_detailed_durations()