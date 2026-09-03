import cv2
import av
import queue
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Counting Sparepart", layout="centered")
st.title("Sistem Counting Sparepart Online")
st.write("Tekan tombol **START** di bawah untuk menyalakan kamera HP/Laptop Anda.")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Konfigurasi Server RTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 1. PERBAIKAN: Simpan queue di session_state agar tidak terhapus saat halaman me-refresh
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Tambahkan conf=0.25 agar deteksi lebih peka saat merekam objek
    results = model.predict(img, conf=0.25)
    annotated_img = results[0].plot()
    
    count = len(results[0].boxes)
    
    # 2. Kirim angka ke antrean permanen
    st.session_state.result_queue.put(count)
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

ctx = webrtc_streamer(
    key="deteksi-sparepart",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("---") 
count_placeholder = st.empty()
count_placeholder.write("Jumlah: 0")

# 3. Ambil data secara real-time dari queue permanen
if ctx and ctx.state.playing:
    while True:
        try:
            count = st.session_state.result_queue.get(timeout=1.0)
            count_placeholder.write(f"Jumlah: {count}")
        except queue.Empty:
            pass