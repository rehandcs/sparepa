import cv2
import av
import queue
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Counting Sparepart", layout="centered")
st.title("Counting")
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

# 1. Buat antrean (queue) untuk mengirim data dari thread video ke UI Streamlit
result_queue = queue.Queue()

def video_frame_callback(frame):
    # Ubah format video
    img = frame.to_ndarray(format="bgr24")
    
    # Deteksi YOLO
    results = model(img)
    annotated_img = results[0].plot()
    
    # Hitung jumlah sparepart
    count = len(results[0].boxes)
    
    # 2. Masukkan hasil perhitungan ke dalam queue
    result_queue.put(count)
    
    # Teks di dalam video (opsional, bisa dihapus jika tidak perlu)
    cv2.putText(annotated_img, f"Jumlah: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Menampilkan WebRTC
ctx = webrtc_streamer(
    key="deteksi-sparepart",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 3. Buat placeholder (wadah kosong) di bawah kamera
count_placeholder = st.empty()

# 4. Ambil data dari queue secara terus-menerus selama kamera menyala
if ctx.state.playing:
    while True:
        try:
            # Ambil nilai count dari antrean (dengan timeout agar tidak memblokir sistem)
            count = result_queue.get(timeout=1.0)
            
            # Perbarui teks di bawah kamera
            count_placeholder.markdown(f"**Count : {count}**")
        except queue.Empty:
            # Jika antrean kosong, abaikan dan coba lagi
            pass