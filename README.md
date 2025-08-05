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
```

---

## 🧠 GPT 요약 기능 사용법 / How to Use GPT Summary

1. OpenAI API 키를 발급받습니다.  
   Get your API key from [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

2. `.streamlit/secrets.toml` 파일을 만들고 아래처럼 작성합니다:

```toml
[openai]
api_key = "sk-..."
```

또는 Streamlit Cloud에서 Secrets 설정에 추가하세요.

3. 앱을 실행하고 체크박스 `🧠 GPT 요약 보기`를 선택하면 자동으로 분석 결과가 생성됩니다.

---

## 🚀 Run Locally / 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Files

| 파일 | 설명 |
|------|------|
| `app.py` | 메인 앱 파일 / Main Streamlit app |
| `cnn_fashion_model.keras` | 학습된 CNN 모델 / Trained CNN model |
| `.gitignore` | 민감한 파일 제외 설정 / Git ignore rules |
| `requirements.txt` | 설치 패키지 목록 / Python dependencies |
| `.streamlit/secrets.toml` | OpenAI 키 설정 / OpenAI API key |
| `README.md` | 프로젝트 설명 문서 / This file |

---

## 📦 Deployment (Streamlit Cloud)

- Streamlit Cloud에 업로드하고, `.streamlit/secrets.toml` 또는 Secrets 설정에서 API 키를 등록하세요.
- You can deploy for free at [https://streamlit.io/cloud](https://streamlit.io/cloud)

---

## 🙋‍♀️ Author

**Suhwa Seong**  
M.Sc. Data Science Student, UE Germany  
GitHub: [https://github.com/SuhwaSeong](https://github.com/SuhwaSeong)
