import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="AI Labeller Pro", layout="wide")

# --- CONFIGURACIÓN DE ETIQUETAS ---
QUICK_LABELS = {
    "1": "excavando",
    "2": "cargando",
    "3": "movimiento",
    "4": "descargando",
    "5": "idle",
}

st.title("⚡ AI Video Labeller Ultra-Fast")

DATA_DIR = "data"
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "video_annotations.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# --- SIDEBAR: CONFIG Y HISTORIAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    video_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".mp4")]
    selected_video = st.selectbox("Seleccione Video", video_files)

    st.divider()
    st.header("📋 Historial (Últimas 15)")
    if os.path.exists(ANNOTATIONS_FILE):
        df_hist = pd.read_csv(ANNOTATIONS_FILE)
        # Mostrar solo las del video actual
        df_video = df_hist[df_hist["video"] == selected_video].tail(15)
        st.dataframe(
            df_video[["start_time", "end_time", "label"]], use_container_width=True
        )
    else:
        st.info("Sin anotaciones aún.")

if not selected_video:
    st.stop()

video_path = os.path.join(DATA_DIR, selected_video)

# --- PANEL PRINCIPAL ---
col_vid, col_ctrl = st.columns([3, 1])

with col_vid:
    st.video(video_path)

with col_ctrl:
    st.markdown("### ⌨️ Atajos de Teclado")
    st.code(
        """
Espacio : Play / Pause
I       : Marcar IN
O       : Marcar OUT
A / D   : -5s / +5s
1 - 5   : Label & Save
    """,
        language="text",
    )

    st.divider()
    # Inputs que el JS rellenará
    t_in = st.text_input("🚩 Inicio (s)", value="0.0", key="t_in")
    t_out = st.text_input("🏁 Fin (s)", value="0.0", key="t_out")
    label_input = st.text_input("🏷️ Etiqueta Actual", key="label_input")

    if st.button("💾 Guardar Segmento (Enter)", use_container_width=True):
        if label_input:
            new_data = {
                "video": selected_video,
                "start_time": t_in,
                "end_time": t_out,
                "label": label_input,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            df = (
                pd.read_csv(ANNOTATIONS_FILE)
                if os.path.exists(ANNOTATIONS_FILE)
                else pd.DataFrame()
            )
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(ANNOTATIONS_FILE, index=False)
            st.toast(f"✅ Guardado: {label_input}")
            # Forzamos un pequeño refresco para actualizar el sidebar
            st.rerun()

# --- PUENTE JAVASCRIPT ---
st.components.v1.html(
    f"""
<script>
const doc = window.parent.document;

function setVal(label, val) {{
    const input = doc.querySelector('input[aria-label="'+label+'"]');
    if (input) {{
        input.value = val;
        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}
}}

function clickSave() {{
    const btns = Array.from(doc.querySelectorAll('button p'));
    const saveBtn = btns.find(p => p.innerText.includes("Guardar Segmento"));
    if (saveBtn) saveBtn.click();
}}

doc.addEventListener('keydown', function(e) {{
    const video = doc.querySelector('video');
    if (!video) return;
    
    if (e.target.tagName === 'INPUT') return;

    const key = e.key.toLowerCase();

    if (e.code === "Space") {{
        e.preventDefault();
        video.paused ? video.play() : video.pause();
    }} 
    else if (key === "i") {{
        setVal("🚩 Inicio (s)", video.currentTime.toFixed(2));
    }} 
    else if (key === "o") {{
        setVal("🏁 Fin (s)", video.currentTime.toFixed(2));
    }}
    else if (key === "a") {{
        video.currentTime = Math.max(0, video.currentTime - 5);
    }}
    else if (key === "d") {{
        video.currentTime = Math.min(video.duration, video.currentTime + 5);
    }}
    
    const labels = {str(QUICK_LABELS)};
    if (labels[e.key]) {{
        setVal("🏁 Fin (s)", video.currentTime.toFixed(2));
        setVal("🏷️ Etiqueta Actual", labels[e.key]);
        setTimeout(clickSave, 100);
    }}
}});
</script>
""",
    height=0,
)
