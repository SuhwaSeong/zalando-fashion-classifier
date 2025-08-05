# 👚 Zalando Fashion Classifier

👠 딥러닝 모델로 Zalando 스타일의 패션 이미지를 분류하는 Streamlit 앱입니다.  
This is a bilingual (🇰🇷/🇺🇸) Streamlit web app that classifies Zalando-style fashion images using a CNN model trained on Fashion MNIST.

---

## 📌 Features / 기능

- ✅ CNN 기반 패션 이미지 분류 / CNN-based image classification  
- 🎨 예측 결과에 따라 이미지 테두리 색상 표시 / Colored borders based on prediction  
- 📊 혼동 행렬 & 클래스별 정확도 시각화 / Confusion matrix & per-class accuracy  
- 🧠 GPT-4 기반 모델 성능 자동 요약 / GPT-4 summary of model performance  
- 💾 예측 결과 저장 및 CSV 다운로드 / Save & download prediction history  
- 🖼️ 최근 업로드 이미지 갤러리 / Recent uploaded image gallery  
- 🌐 한글 + 영어 병기 UI / Korean-English bilingual UI  

---

## 🖼️ Sample Screenshots

📌 *Insert screenshot image files under a folder named `/screenshots` in your repo.*  
예시 이미지 파일을 `/screenshots` 폴더에 넣어주세요.

```bash
/screenshots/example1.png
/screenshots/example2.png
````

---

## 📂 Dataset Source

* This app uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) provided by Zalando Research.
* It contains 28x28 grayscale images of 10 fashion categories such as T-shirt/top, Trouser, Sneaker, etc.

---

## 🧠 Model Training Info

* The CNN model was trained using TensorFlow/Keras on the Fashion MNIST dataset.
* Training was performed for 10 epochs with accuracy around 89%.
* You can optionally retrain the model using a custom script (`train_model.py`, not included by default).

---

## 🧠 GPT 요약 기능 사용법 / How to Use GPT Summary

📌 이 기능은 저장된 예측 결과를 바탕으로 GPT가 모델의 성능을 분석 및 요약해주는 기능입니다.
📌 This feature uses GPT to summarize your model performance based on saved predictions.

### ✅ 사용 조건 / Requirements:

* 최소 **2개 이상의 예측 결과**가 저장되어 있어야 합니다.
  At least **2 predictions** must be saved.
* 예측 후 **"예측 저장" 버튼**을 눌러야 데이터가 기록됩니다.
  You must click **"Save Prediction"** after prediction to store the result.

### 📤 출력 내용 / Output Includes:

* 전체 예측 수 / Total predictions
* 정확도 / Overall accuracy
* 클래스별 예측 분포 / Class-wise prediction distribution
* GPT가 자동 생성한 요약 (영문) / GPT-generated summary (in English)

> 🔐 GPT 요약은 OpenAI API를 통해 실행됩니다.
> *GPT summary is powered by OpenAI API. Ensure your API key is configured properly in Streamlit Secrets.*

---

## 🚀 Run Locally / 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Files

| 파일                        | 설명                              |
| ------------------------- | ------------------------------- |
| `app.py`                  | 메인 앱 파일 / Main Streamlit app    |
| `cnn_fashion_model.keras` | 학습된 CNN 모델 / Trained CNN model  |
| `.gitignore`              | 민감한 파일 제외 설정 / Git ignore rules |
| `requirements.txt`        | 설치 패키지 목록 / Python dependencies |
| `.streamlit/secrets.toml` | OpenAI 키 설정 / OpenAI API key    |
| `README.md`               | 프로젝트 설명 문서 / This file          |

---

## 📦 Deployment (Streamlit Cloud)

* Streamlit Cloud에 업로드하고, `.streamlit/secrets.toml` 또는 Secrets 설정에서 OpenAI API 키를 등록하세요.
* You can deploy for free at [https://streamlit.io/cloud](https://streamlit.io/cloud)

---

## 🙋‍♀️ Author

**Suhwa Seong**  
M.Sc. Data Science Student, UE Germany  
GitHub: [https://github.com/SuhwaSeong](https://github.com/SuhwaSeong)

```
