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

# 1. Buat antrean (queue) untuk mengirim data dari video ke UI Streamlit
result_queue = queue.Queue()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    results = model(img)
    annotated_img = results[0].plot()
    
    count = len(results[0].boxes)
    
    # 2. Kirim nilai count ke antrean setiap kali frame diproses
    result_queue.put(count)
    
    # (Opsional) Tetap tampilkan tulisan di dalam video
    cv2.putText(annotated_img, f"Jumlah: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

ctx = webrtc_streamer(
    key="deteksi-sparepart",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 3. Siapkan placeholder di frontend (di bawah letak kamera Streamlit)
st.write("---") # Garis pemisah agar rapi
count_placeholder = st.empty()

# 4. Tampilkan "Jumlah: 0" sebagai default jika kamera menyala tapi belum ada deteksi
if ctx.state.playing:
    count_placeholder.write("### Jumlah : 0")
    
    # Ambil data dari antrean terus-menerus
    while True:
        try:
            count = result_queue.get(timeout=1.0)
            # 5. Timpa nilai sebelumnya dengan nilai baru menggunakan format st.write()
            count_placeholder.write(f"### Jumlah : {count}")
        except queue.Empty:
            pass