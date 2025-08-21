import streamlit as st 
import time

# Jey de la vie
from utils_life_rules import evolution, format_BS_rule_to_inter_list

# web rendering
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils_rendering import GoLEngine, GoLParams
import av



st.set_page_config('Life', layout='wide', page_icon="🧬")
st.title("🧬 Jeu de la vie")
st.divider()

# paramètres dans la sidebar
with st.sidebar:
    st.header("Paramètres")
    rows = st.slider("Lignes", 10, 300, 60, step=2)
    cols = st.slider("Colonnes", 10, 400, 100, step=2)
    cell_size = st.slider("Taille de cellule (px)", 2, 20, 8, step=1)
    fps = st.slider("SPS (Etats/sec)", 1, 60, 10, step=1)
    periodic = st.checkbox("Bords périodiques (torus)", True)
    rule = st.text_input("Règle B/S", "B3/S23")
    life_density = st.slider("Densité initiale de vie", 0.0, 1.0, 0.2, step=0.05)


# état partagé
if "gol_engine" not in st.session_state:
    st.session_state.gol_engine = GoLEngine(
        GoLParams(rows=rows, cols=cols, rule=rule, periodic=periodic,
                  life_density=life_density, fps=fps, cell_size=cell_size)
    )

engine: GoLEngine = st.session_state.gol_engine
# mettre à jour les params live (taille nécessite reset pour prendre effet)
engine.update_params(rule=rule, periodic=periodic, fps=fps, cell_size=cell_size, life_density=life_density)


# ----------- WebRTC -----------
# Le callback génère nos frames côté serveur (pas d'entrée caméra).

_, col1, _, col2, _ = st.columns((1, 7, 1 ,20, 1))

with col1:
        cont_1 = st.container(border = True)
        with cont_1:
            B, S = format_BS_rule_to_inter_list(BS_rule=rule)
            st.markdown(
                            f"""
                            **Règle pour chaque cellule :** 
                            - 🐣 Elle nait à coté de {(', '.join(str(b) for b in B))} cellule(s).
                            - 💚 Elle survit à coté de {(', '.join(str(b) for b in S))} cellule(s).
                            - ☠️ Elle meurt dans les autres cas
                            """,
            unsafe_allow_html=True
        )
            
        btn_start = st.button("Start", use_container_width=True)
        btn_stop = st.button("Stop", use_container_width=True)
        btn_reset = st.button("Reset", use_container_width=True)
        cont_2 = st.container(border=True)
        with cont_2:
            btn_pause = st.toggle("Pause", value=False)
            btn_step = st.button("Step +1")

if btn_start and not engine.running:
    engine.toggle_running(True)
    
if btn_stop:
    engine.toggle_running(False)
    
if btn_reset or (engine.params.rows != rows or engine.params.cols != cols):
    engine.update_params(rows=rows, cols=cols)
    engine.reset()

#engine.toggle_running(not btn_pause)

# un “step” manuel si en pause
if btn_step and btn_pause:
    engine.toggle_running(False)
    with engine.lock:
        engine.grid = evolution(engine.grid, rule=engine.params.rule, periodic_border=engine.params.periodic)


with col2:
    placeholder = st.empty()
    while engine.running:
        img = engine.step_and_render_bgr()
        placeholder.image(img, channels="BGR", use_container_width=True)
        time.sleep(1.0 / fps)
        
    

if False:
    def video_frame_callback():
        img_bgr = engine.step_and_render_bgr()
        # rythme
        time.sleep(1.0 / max(1, engine.params.fps))
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    webrtc_streamer(
        key="gol-recvonly",
        mode=WebRtcMode.RECVONLY,                 # ✅ le client ne capture rien, il reçoit seulement
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": False, "audio": False},  # ✅ pas d’accès webcam/micro
        rtc_configuration={                       # (optionnel) STUN public pour passer les NAT
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )
