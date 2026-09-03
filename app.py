import cv2
import av
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

# Konfigurasi Server RTC (Agar streaming video lancar di jaringan publik)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Fungsi untuk memproses setiap frame video dari HP pengguna
def video_frame_callback(frame):
    # Ubah format video dari web ke format gambar OpenCV
    img = frame.to_ndarray(format="bgr24")
    
    # Deteksi YOLO
    results = model(img)
    annotated_img = results[0].plot()
    
    # Hitung jumlah sparepart
    count = len(results[0].boxes)
    
    # Tulis teks jumlah di pojok kiri atas gambar
    cv2.putText(annotated_img, f"Jumlah: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Kembalikan gambar ke browser pengguna
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Menampilkan antarmuka WebRTC (Tombol Start/Stop otomatis dari library ini)
webrtc_streamer(
    key="deteksi-sparepart",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)