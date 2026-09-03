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

# Buat antrean (queue) untuk mengirim data jumlah dari video ke UI Streamlit
result_queue = queue.Queue()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    results = model(img)
    annotated_img = results[0].plot()
    
    # Hitung jumlah sparepart yang terdeteksi
    count = len(results[0].boxes)
    
    # Kirim angka hitungan ke antrean
    result_queue.put(count)
    
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

# 1. Siapkan wadah di bawah kamera
count_placeholder = st.empty()

# 2. Tampilkan teks default DI LUAR if, agar selalu muncul di HP sejak awal
count_placeholder.write("Jumlah: 0")

# 3. Update angka hanya jika kamera sudah benar-benar menyala dan diizinkan browser
if ctx and ctx.state.playing:
    while True:
        try:
            count = result_queue.get(timeout=1.0)
            count_placeholder.write(f"Jumlah: {count}")
        except queue.Empty:
            pass