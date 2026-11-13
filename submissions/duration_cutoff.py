"""Berechnet die Mietdauer-Verteilung, setzt einen 90%-Cutoff 
und kappt alle längeren Mieten auf diesen Tag, um die Matrix klein zu halten.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

def analyze_and_cap_durations():
    csv_path = HERE / "data" / "reservations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")
        
    df = pd.read_csv(csv_path, parse_dates=["pickup_date", "return_date"])
    
    # Nur abgeschlossene Buchungen betrachten
    df = df[df["status"] == "completed"].copy()
    
    # 1. Exakte Mietdauer in Tagen berechnen
    df["duration"] = (df["return_date"] - df["pickup_date"]).dt.days + 1
    df["duration"] = df["duration"].apply(lambda x: max(1, x))
    
    detailed_rows = []
    
    # 2. Pro Station und Kategorie auswerten
    grouped = df.groupby(["pickup_station_id", "category"])
    
    for (station, category), group in grouped:
        durations = group["duration"].copy()
        total_count = len(durations)
        
        if total_count == 0:
            continue
            
        # Temporäre Häufigkeiten berechnen, um den 90%-Cutoff zu finden
        temp_counts = durations.value_counts().sort_index()
        temp_probs = temp_counts / total_count
        temp_cum_probs = temp_probs.cumsum()
        
        # Finde den Tag, ab dem die kumulierte Wahrscheinlichkeit >= 90% ist
        cutoff_day = None
        for duration, cum_p in temp_cum_probs.items():
            if cum_p >= 0.90:
                cutoff_day = duration
                break
        
        # Fallback falls die Stichprobe zu klein ist
        if cutoff_day is None:
            cutoff_day = durations.max()
            
        # --- DER TRICK: Alles was > cutoff_day ist, wird auf cutoff_day gesetzt ---
        durations = durations.apply(lambda d: min(d, cutoff_day))
        
        # Jetzt die finale Verteilung mit den gecappten Dauern berechnen
        counts = durations.value_counts().sort_index()
        
        for duration, count in counts.items():
            share = count / total_count
            detailed_rows.append({
                "station_id": station,
                "category": category,
                "duration_days": duration,
                "count": count,
                "share_percent": round(share * 100, 2),
                "is_cutoff_cap": (duration == cutoff_day) # Markiert den Tag, wo der Rest draufgeschlagen wurde
            })
            
    res_df = pd.DataFrame(detailed_rows)
    
    # Kumulierte Wahrscheinlichkeit berechnen
    res_df["cum_prob_percent"] = (
        res_df.groupby(["station_id", "category"])["share_percent"]
        .cumsum()
    )
    
    # Speichern
    out_path = HERE / "duration_distribution_capped.csv"
    res_df.to_csv(out_path, index=False)
    
    print("--- Gecappte Mietdauer-Verteilung (Max Cutoff bei 90%) ---")
    print(res_df.head(15).to_string(index=False))
    print(f"\nGecappte Tabelle gespeichert unter: {out_path}")

if __name__ == "__main__":
    analyze_and_cap_durations()