import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image, ImageOps
import os
import openai

# 클래스 레이블 정의 / Define class labels
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 모델 로드 / Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_fashion_model.keras")

model = load_model()

# 테두리 색상 / Set border color by predicted class
border_colors = {
    "T-shirt/top": "gray", "Trouser": "olive", "Pullover": "purple",
    "Dress": "pink", "Coat": "brown", "Sandal": "orange",
    "Shirt": "teal", "Sneaker": "blue", "Bag": "green", "Ankle boot": "red"
}

# UI 시작 / App UI start
st.title("👚 Zalando 패션 이미지 분류기 / Zalando Fashion Classifier")
st.write("Zalando 스타일의 패션 이미지를 업로드하세요.\nUpload a fashion image to classify it.")

uploaded_file = st.file_uploader("📁 이미지 업로드 / Upload image", type=["png", "jpg", "jpeg"])
recent_folder = "recent_uploads"
os.makedirs(recent_folder, exist_ok=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)[0]
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    color = border_colors.get(predicted_class, "black")
    bordered_image = ImageOps.expand(image, border=8, fill=color)
    st.image(bordered_image, caption=f"🎯 예측 / Predicted: {predicted_class}", use_container_width=True)

    st.subheader(f"🔍 예측 결과 / Prediction: {predicted_class}")
    st.write(f"Confidence (신뢰도): {confidence * 100:.2f}%")
    fig, ax = plt.subplots()
    ax.barh(labels, prediction)
    ax.set_xlabel("Probability / 예측 확률")
    st.pyplot(fig)

    actual_label = st.selectbox("✅ 실제 레이블을 선택하세요 / Select the actual label", labels)

    if st.button("📏 예측 저장 / Save Prediction"):
        csv_path = "fashion_predictions.csv"
        new_data = pd.DataFrame([{
            "filename": uploaded_file.name,
            "predicted": predicted_class,
            "actual": actual_label
        }])
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            updated = pd.concat([existing, new_data], ignore_index=True)
        else:
            updated = new_data
        updated.to_csv(csv_path, index=False)
        st.success("✅ 예측이 저장되었습니다 / Prediction saved!")

    recent_path = os.path.join(recent_folder, uploaded_file.name)
    bordered_image.save(recent_path)
    st.success("📅 최근 업로드에 이미지 저장됨 / Image saved to recent uploads.")

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('예측 / Predicted')
    ax.set_ylabel('실제 / Actual')
    ax.set_title('Confusion Matrix / 혼동 행렬')
    st.pyplot(fig)

def plot_class_accuracy(y_true, y_pred, class_names):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    class_acc = df.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))
    fig, ax = plt.subplots()
    class_acc[class_names].plot(kind='bar', ax=ax)
    ax.set_title("클래스별 정확도 / Class-wise Accuracy")
    ax.set_ylabel("정확도 / Accuracy")
    st.pyplot(fig)

st.markdown("---")

if st.checkbox("📊 혼동 행렬 및 정확도 보기 / Show Confusion Matrix & Accuracy"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        y_true = df["actual"]
        y_pred = df["predicted"]
        col1, col2 = st.columns(2)
        with col1:
            plot_confusion_matrix(y_true, y_pred, labels)
        with col2:
            plot_class_accuracy(y_true, y_pred, labels)
    except Exception as e:
        st.error(f"❌ 오류 발생 / Error loading prediction file: {e}")

if st.checkbox("⬇️ 예측 기록 다운로드 / Download Prediction CSV"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📁 CSV 다운로드 / Download CSV", csv, "fashion_predictions.csv", "text/csv")
    except Exception as e:
        st.warning(f"다운로드 실패 / CSV download failed: {e}")

if st.checkbox("🖼️ 최근 업로드 이미지 보기 / Show Recent Uploads"):
    files = os.listdir(recent_folder)
    if files:
        cols = st.columns(min(5, len(files)))
        for i, file in enumerate(files[:10]):
            img = Image.open(os.path.join(recent_folder, file))
            cols[i % len(cols)].image(img, caption=file, use_container_width=True)
    else:
        st.info("⚠️ 업로드된 이미지가 없습니다 / No images found.")

openai.api_key = st.secrets["openai"]["api_key"]

with st.expander("📘 GPT 요약 기능 사용법 / How to Use GPT-based Summary"):
    st.markdown("""
### 🧐 GPT 요약 기능 안내 (Korean)

- 이 기능은 OpenAI의 GPT-4를 사용해서 예측 결과를 자동으로 분석해주는 기능입니다.
- `fashion_predictions.csv` 파일에 최소 2개 이상의 예측 결과가 포함되어야 합니다.
- GPT를 사용하려면 OpenAI API 키가 필요합니다.

#### 사용 방법:
1. [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) 에서 API 키를 발급 받으세요.
2. `secrets.toml` 또는 Streamlit Cloud의 **Secrets Settings**에 아래와 같이 저장하세요:
```toml
[openai]
api_key = "sk-..."
```
3. 앱을 실행하고 '🧐 GPT 요약 보기' 체크박스를 선택하세요.

### 🧐 GPT Summary Instructions (English)
- This feature uses OpenAI GPT-4 to summarize model performance based on prediction results.
- You must have at least 2 records in fashion_predictions.csv.
- An OpenAI API key is required.

How to use:
Get your key from https://platform.openai.com/account/api-keys

Store it in .streamlit/secrets.toml or in Streamlit Cloud → Settings → Secrets:
```toml
[openai]
api_key = "sk-..."
```
3. Check the box "🧐 GPT-based Model Summary" to view the summary.
""")

# ✅ GPT 요약 실행 / Run GPT-based summary
if st.checkbox("🧐 GPT 요약 보기 / Show GPT-based Model Summary"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        if len(df) < 2:
            st.warning("⚠️ 최소 2개 이상의 예측 결과가 필요합니다 / At least 2 predictions required.")
        else:
            prompt = f"""
You are an expert data analyst. Please summarize the model performance based on the following prediction results:

{df.to_csv(index=False)}

Include insights such as overall accuracy, frequent misclassifications, and class-wise performance.
"""
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.subheader("🧠 GPT 요약 결과 / GPT Summary")
            st.markdown(response.choices[0].message.content)
    except Exception as e:
        st.error(f"❌ GPT 요약 실패 / GPT summary failed: {e}")
""")
