import pandas as pd
import numpy as np
import re
from datetime import datetime

def time_to_sec(t_str):
    try:
        # Manejar formato MM:SS o solo SS
        parts = t_str.strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])
    except:
        return None

def parse_labels(file_path):
    labels = []
    current_offset = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line: continue
            
            # Extraer tiempos y actividad
            match = re.search(r'(\d+:\d+|\d+)\s*-\s*(\d+:\d+|\d+)\s*->\s*(.*)', line)
            if match:
                start = time_to_sec(match.group(1))
                end = time_to_sec(match.group(2))
                activity = match.group(3).strip()
                labels.append({'start': start, 'end': end, 'label': activity})
    
    return pd.DataFrame(labels)

def apply_robust_filtering(df_labels, margin=1.0):
    """
    Crea zonas de exclusión alrededor de cada cambio de etiqueta 
    para no confundir al modelo con el error de 1s del humano.
    """
    robust_labels = []
    for i, row in df_labels.iterrows():
        # Acortar el segmento por el margen en ambos lados
        new_start = row['start'] + margin
        new_end = row['end'] - margin
        
        if new_end > new_start:
            robust_labels.append({
                'start': new_start, 
                'end': new_end, 
                'label': row['label'],
                'original_start': row['start']
            })
    
    return pd.DataFrame(robust_labels)

if __name__ == "__main__":
    print("Analizando labels.txt...")
    raw_labels = parse_labels('labels.txt')
    print(f"Detectados {len(raw_labels)} segmentos manuales.")
    
    robust_df = apply_robust_filtering(raw_labels, margin=1.0)
    print(f"Segmentos robustos (tras quitar margen de 1s): {len(robust_df)}")
    
    robust_df.to_csv('data/refined/labels_robust.csv', index=False)
    print("Guardado en data/refined/labels_robust.csv")
