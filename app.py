import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image

API_KEY = "uPlMOzPR8zCgJqURmNf2"  
WORKSPACE_ID = "kittyyunees-workspace"
PROJECT_ID = "pet-cap-ring-detection"       
MODEL_VERSION = 3                           

IMAGE_SAVE_DIR = "qr_captured_rings"
CSV_FILE = "qr_detection_logs.csv"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Local_Image_Path", "Detected_Count"])
    df.to_csv(CSV_FILE, index=False)

@st.cache_resource
def init_roboflow():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
    model = project.version(MODEL_VERSION).model
    return project, model

project, model = init_roboflow()

st.title("♻️ 캡링 고화질 수집 시스템")
st.write("아이폰/갤럭시 기본 카메라로 선명하게 찍은 사진을 올리면 원래 모델 성능 그대로 완벽하게 인식합니다.")

# 💡 핵심 변경: 흐릿한 실시간 웹캠 대신 고화질 파일 업로더를 메인으로 사용합니다.
uploaded_file = st.file_uploader("📸 폰 카메라로 찍은 선명한 사진을 올려주세요!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. 원본 화질 그대로 이미지 불러오기
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    temp_path = "temp_input.jpg"
    # 화질 저하 없이 원본 퀄리티 그대로 임시 저장
    cv2.imwrite(temp_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    with st.spinner("AI가 고화질 원본 분석 중..."):
        # 💡 신뢰도(Confidence)를 25%로 낮추어 놓치지 않게 설정
        prediction_result = model.predict(temp_path, confidence=25).json()
        predictions = prediction_result.get("predictions", [])
    
    detected_count = len(predictions)
    
    # 2. 결과 시각화 (선 두께와 글씨 크기도 고화질에 맞게 크게 조정)
    h_img, w_img, _ = image_bgr.shape
    scale = max(h_img, w_img) / 1000  # 이미지 크기에 맞게 UI 스케일 조절
    
    for pred in predictions:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), int(6 * scale))
        cv2.putText(image_bgr, f"{pred['class']} {pred['confidence']:.2f}", (x1, y1 - int(15 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 255, 0), int(3 * scale))

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="분석 결과", use_column_width=True)
    
    # 3. 데이터 저장
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"ring_{now}.jpg"
    save_path = os.path.join(IMAGE_SAVE_DIR, save_filename)
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    new_log = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Local_Image_Path": save_path, "Detected_Count": detected_count}])
    new_log.to_csv(CSV_FILE, mode='a', header=False, index=False)
    st.success(f"💾 내 컴퓨터 저장 완료! (검출: {detected_count}개)")
    
    # 4. 로보플로 서버 전송 오타 수정 (`image_path`로 명시)
    with st.spinner("로보플로 클라우드로 자동 전송 중..."):
        try:
            project.upload(image_path=save_path, batch_name="active_learning_stream")
            st.toast("🚀 로보플로 업로드 성공! Auto-Label 완료.")
        except Exception as e:
            st.warning(f"업로드 실패했으나 로컬에는 저장됨: {e}")
            
    if os.path.exists(temp_path):
        os.remove(temp_path)