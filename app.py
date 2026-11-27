import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import json
import os
import requests
import streamlit.components.v1 as components
from PIL import Image
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="FreshFruit AI", page_icon="🍎", layout="wide")

MODEL_PATH = "fruit_detector.keras"
CLASS_PATH = "class_names.json"
HISTORY_PATH = "prediction_history.csv"
CAMERA_API = "http://127.0.0.1:8001"

st.markdown("""
<style>
.stApp{background:#080f1e;color:#f1f5f9}
[data-testid="stSidebar"]{background:#101a2d;border-right:1px solid #24334d}
h1,h2,h3{color:#f8fafc!important}
p,label{color:#cbd5e1}
.metric-card{background:linear-gradient(135deg,#17243b,#101a2d);border:1px solid #263b59;padding:22px;border-radius:16px;margin-bottom:12px}
.metric-title{color:#94a3b8;font-size:14px}
.metric-value{color:#38bdf8;font-size:28px;font-weight:700}
.result-card{background:#13243a;border:1px solid #285477;border-radius:16px;padding:22px;margin-top:15px}
.stButton>button{background:#0891b2;color:white;border:0;border-radius:9px;font-weight:600}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        try:
            return [v for k, v in sorted(data.items(), key=lambda x: int(x[0]))]
        except (ValueError, TypeError):
            return list(data.values())
    raise ValueError("Unsupported class_names.json format")

def load_history():
    cols = ["timestamp", "fruit", "freshness", "confidence", "source"]
    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_csv(HISTORY_PATH)
            if not df.empty:
                for c in cols:
                    if c not in df.columns:
                        df[c] = ""
                return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

def save_prediction(fruit, freshness, confidence, source):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fruit": fruit, "freshness": freshness,
        "confidence": round(float(confidence), 2), "source": source
    }])
    pd.concat([load_history(), row], ignore_index=True).to_csv(HISTORY_PATH, index=False)

def predict_image(image, model, class_names):
    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)
    pred = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(pred))
    predicted_class = class_names[idx]
    confidence = float(pred[idx]) * 100
    lower = predicted_class.lower()
    if lower.startswith("fresh"):
        freshness, fruit = "Fresh", predicted_class[5:]
    elif lower.startswith("rotten"):
        freshness, fruit = "Rotten", predicted_class[6:]
    else:
        freshness, fruit = "Unknown", predicted_class
    return fruit.replace("_", " ").title(), freshness, confidence, predicted_class

st.sidebar.markdown("## 🍎 FreshFruit AI")
st.sidebar.caption("Intelligent Fruit Freshness Detection")

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    st.error("Keep fruit_detector.keras and class_names.json in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"Could not load model/classes: {e}")
    st.stop()

page = st.sidebar.radio("NAVIGATION", [
    "Dashboard", "Fruit Detection", "Prediction History", "Analytics", "Reports"
])
st.sidebar.divider()
st.sidebar.success("AI Model Loaded")
st.sidebar.caption("MobileNetV2 | 224 × 224")
st.sidebar.caption("Real-time webcam uses browser camera + local API")

history = load_history()

if page == "Dashboard":
    st.title("🍎 FreshFruit AI Dashboard")
    st.write("AI-powered fruit freshness detection system.")
    total = len(history)
    fresh = int((history["freshness"] == "Fresh").sum()) if total else 0
    rotten = int((history["freshness"] == "Rotten").sum()) if total else 0
    avg = float(pd.to_numeric(history["confidence"], errors="coerce").mean()) if total else 0
    cols = st.columns(4)
    metrics = [(cols[0], "Total Predictions", total),
               (cols[1], "Fresh Fruits", fresh),
               (cols[2], "Rotten Fruits", rotten),
               (cols[3], "Avg. Confidence", f"{avg:.2f}%")]
    for col, title, value in metrics:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.subheader("Recent Predictions")
    if total:
        st.dataframe(history.tail(8).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Open Fruit Detection to get started.")
    st.subheader("About the AI System")
    st.write("MobileNetV2 classifies images into eight trained fresh/rotten fruit classes.")
    st.write(", ".join(class_names))

elif page == "Fruit Detection":
    st.title("🔍 AI Fruit Detection")
    mode = st.radio("Detection mode", ["Live Webcam", "Upload Image"], horizontal=True)
    if mode == "Live Webcam":
        st.info("Allow camera access in your browser. The webcam frames are sent to your local Python API for prediction.")
        try:
            r = requests.get(f"{CAMERA_API}/health", timeout=2)
            api_ok = r.ok
        except requests.RequestException:
            api_ok = False
        if not api_ok:
            st.error("Live camera API is not running. Start camera_api.py in a second terminal (instructions below).")
            st.code("python -m uvicorn camera_api:app --host 127.0.0.1 --port 8001")
        else:
            components.html("""
<!DOCTYPE html><html><head><style>
body{margin:0;background:#0b1527;color:#e5edf8;font-family:Arial}
.wrap{background:#101d32;border:1px solid #29415f;border-radius:14px;padding:14px}
video{width:100%;max-height:430px;background:#050a12;border-radius:10px;transform:scaleX(-1)}
.controls{display:flex;gap:10px;margin:12px 0;flex-wrap:wrap}
button{border:0;border-radius:8px;padding:10px 16px;font-weight:bold;cursor:pointer}
#start{background:#0891b2;color:white}#stop{background:#334155;color:white}
#status{color:#93c5fd;font-size:13px;margin:8px 0}
#result{background:#172943;border:1px solid #315477;border-radius:10px;padding:14px;margin-top:10px}
#label{font-size:25px;font-weight:bold;color:#38bdf8}
#confidence{color:#cbd5e1;margin-top:5px}
small{color:#94a3b8}
</style></head><body><div class="wrap">
<h3>📷 Live Fruit Detection</h3>
<video id="video" autoplay playsinline muted></video>
<canvas id="canvas" width="224" height="224" style="display:none"></canvas>
<div class="controls"><button id="start">▶ Start Camera</button><button id="stop" disabled>■ Stop Camera</button>
<label style="display:flex;align-items:center;gap:7px;color:#cbd5e1"><input id="voiceOn" type="checkbox" checked> 🔊 Voice Assistant</label></div>
<div id="status">Camera is stopped.</div>
<div id="result"><div id="label">Waiting for camera...</div><div id="confidence"></div></div>
<small>Predictions update approximately once per second. Camera frames stay on this computer.</small>
</div>
<script>
const video=document.getElementById('video'), canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d'), statusEl=document.getElementById('status');
const label=document.getElementById('label'), confidence=document.getElementById('confidence');
const start=document.getElementById('start'), stop=document.getElementById('stop');
const voiceOn=document.getElementById('voiceOn');
let stream=null, timer=null, busy=false, lastKey="", lastSavedAt=0;
function speakPrediction(text){
 if(!voiceOn.checked || !('speechSynthesis' in window)) return;
 window.speechSynthesis.cancel();
 const utterance=new SpeechSynthesisUtterance(text);
 utterance.lang='en-IN'; utterance.rate=0.95; utterance.pitch=1;
 window.speechSynthesis.speak(utterance);
}
async function tick(){
 if(!stream || busy || video.readyState<2) return;
 busy=true;
 try{
  ctx.drawImage(video,0,0,224,224);
  const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',0.75));
  const form=new FormData(); form.append('file',blob,'frame.jpg');
  const response=await fetch('http://127.0.0.1:8001/predict',{method:'POST',body:form});
  if(!response.ok) throw new Error('Prediction API error');
  const data=await response.json();
  const key=data.predicted_class;
  if(key!==lastKey){
    speakPrediction(data.fruit+' detected. '+data.freshness+' fruit. Confidence '+data.confidence.toFixed(0)+' percent.');
  }
  label.textContent=data.fruit.toUpperCase()+' — '+data.freshness.toUpperCase();
  label.style.color=data.freshness==='Fresh'?'#4ade80':(data.freshness==='Rotten'?'#fb7185':'#38bdf8');
  confidence.textContent='Confidence: '+data.confidence.toFixed(2)+'%';
  statusEl.textContent='● LIVE — scanning continuously';
  const now=Date.now();
  if(key!==lastKey && now-lastSavedAt>5000){lastKey=key;lastSavedAt=now;}
 }catch(e){statusEl.textContent='Prediction issue: '+e.message;}
 finally{busy=false;}
}
start.onclick=async()=>{
 try{
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
  video.srcObject=stream;
  await video.play();
  start.disabled=true;stop.disabled=false;
  statusEl.textContent='Camera started. Analyzing frames...';
  timer=setInterval(tick,1000);
 }catch(e){statusEl.textContent='Camera access failed: '+e.message+'. Use localhost and allow camera permission.';}
};
stop.onclick=()=>{
 if(timer)clearInterval(timer);timer=null;
 if(stream)stream.getTracks().forEach(t=>t.stop());
 stream=null;video.srcObject=null;start.disabled=false;stop.disabled=true;
 statusEl.textContent='Camera stopped.';
 label.textContent='Waiting for camera...';confidence.textContent='';
};
window.addEventListener('beforeunload',()=>{if(timer)clearInterval(timer);if(stream)stream.getTracks().forEach(t=>t.stop());});
</script></body></html>
            """, height=650, scrolling=False)
    else:
        uploaded = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Input image", use_container_width=True)
            if st.button("Analyze Fruit", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    fruit, freshness, confidence, predicted_class = predict_image(image, model, class_names)
                save_prediction(fruit, freshness, confidence, "Upload")
                color = "#22c55e" if freshness == "Fresh" else "#ef4444"
                st.markdown(f'<div class="result-card"><h2>{fruit}</h2><h3 style="color:{color}">{freshness} Fruit</h3><p>Confidence: {confidence:.2f}%</p><p>Class: {predicted_class}</p></div>', unsafe_allow_html=True)
                st.progress(min(max(confidence / 100, 0), 1))
                spoken_text = f"{fruit} detected. {freshness} fruit. Confidence {confidence:.0f} percent."
                import html
                safe_text = html.escape(spoken_text, quote=True)
                components.html(f"""
                <button id="speakResult" style="background:#0891b2;color:white;border:0;border-radius:8px;padding:10px 16px;font-weight:600;cursor:pointer">🔊 Speak Prediction</button>
                <script>
                document.getElementById('speakResult').onclick=()=>{{
                    if(!('speechSynthesis' in window)){{alert('Speech is not supported in this browser.');return;}}
                    window.speechSynthesis.cancel();
                    const msg=new SpeechSynthesisUtterance("{safe_text}");
                    msg.lang='en-IN';msg.rate=0.95;window.speechSynthesis.speak(msg);
                }};
                </script>
                """, height=55)
    st.caption("Model confidence is not a guarantee of food safety.")

elif page == "Prediction History":
    st.title("📋 Prediction History")
    history = load_history()
    if not history.empty:
        st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Prediction History", history.to_csv(index=False).encode("utf-8"), "fruit_prediction_history.csv", "text/csv")
    else:
        st.info("No prediction history available.")

elif page == "Analytics":
    st.title("📊 Analytics Dashboard")
    history = load_history()
    if not history.empty:
        c1, c2 = st.columns(2)
        freshness = history["freshness"].value_counts().rename_axis("Freshness").reset_index(name="Count")
        c1.plotly_chart(px.pie(freshness, names="Freshness", values="Count", title="Fresh vs Rotten", hole=0.5), use_container_width=True)
        fruits = history["fruit"].value_counts().rename_axis("Fruit").reset_index(name="Count")
        c2.plotly_chart(px.bar(fruits, x="Fruit", y="Count", title="Fruit Detection Count"), use_container_width=True)
        history["confidence"] = pd.to_numeric(history["confidence"], errors="coerce")
        st.plotly_chart(px.histogram(history, x="confidence", nbins=10, title="Confidence Distribution"), use_container_width=True)
    else:
        st.info("Make predictions to generate analytics.")

elif page == "Reports":
    st.title("📁 Downloadable Reports")
    history = load_history()
    if not history.empty:
        st.success("Prediction data is ready.")
        st.download_button("⬇ Download CSV Report", history.to_csv(index=False).encode("utf-8"), "FreshFruit_AI_Report.csv", "text/csv", use_container_width=True)
        confidence = pd.to_numeric(history["confidence"], errors="coerce")
        summary = f"""FRESHFRUIT AI - PREDICTION REPORT
Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
Total Predictions: {len(history)}
Fresh Predictions: {(history['freshness'] == 'Fresh').sum()}
Rotten Predictions: {(history['freshness'] == 'Rotten').sum()}
Average Confidence: {confidence.mean():.2f}%
Model: MobileNetV2
Input Size: 224 x 224
"""
        st.download_button("⬇ Download Summary Report", summary, "FreshFruit_AI_Summary.txt", "text/plain", use_container_width=True)
    else:
        st.warning("No prediction data available.")

st.sidebar.divider()
st.sidebar.caption("FreshFruit AI | Powered by MobileNetV2")
st.sidebar.caption("AI-based freshness classification")
