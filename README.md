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

## 🧠 GPT 요약 기능 사용법 / How to Use GPT Summary

1. **로컬에서 실행하는 경우에만 OpenAI API 키가 필요합니다.**
   *(If you run the app locally, you need an API key. If you're using Streamlit Cloud with `Secrets`, skip this step.)*

2. 로컬 실행 시, 프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 아래처럼 작성합니다:

```toml
[openai]
api_key = "sk-..."  # 본인의 OpenAI API 키 입력
```

> 🛑 `.streamlit/secrets.toml` 파일은 **절대 GitHub에 업로드하지 마세요.**

3. Streamlit Cloud에서는 **Settings > Secrets**에 이미 키를 입력한 경우, 추가 설정은 필요하지 않습니다.

4. 앱을 실행하고 `🧠 GPT 기반 모델 요약 보기 / Show GPT-based Model Summary` 체크박스를 선택하면 자동 분석이 실행됩니다.

---

## 🗃️ Original Dataset

* **Fashion MNIST**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
  Zalando에서 제공한 흑백 28x28 픽셀 패션 이미지 데이터셋
  (10개 카테고리: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

---

## 🚀 Run Locally / 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Files

| 파일                        | 설명                                                 |
| ------------------------- | -------------------------------------------------- |
| `app.py`                  | 메인 앱 파일 / Main Streamlit app                       |
| `cnn_fashion_model.keras` | 학습된 CNN 모델 / Trained CNN model                     |
| `.gitignore`              | 민감한 파일 제외 설정 / Git ignore rules                    |
| `requirements.txt`        | 설치 패키지 목록 / Python dependencies                    |
| `.streamlit/secrets.toml` | OpenAI 키 설정 (로컬용) / OpenAI API key (for local use) |
| `README.md`               | 프로젝트 설명 문서 / This file                             |

---

## 📦 Deployment (Streamlit Cloud)

* Streamlit Cloud에서 앱을 업로드하고 **Settings → Secrets**에서 API 키를 입력하세요.
* 무료로 배포할 수 있습니다: [https://streamlit.io/cloud](https://streamlit.io/cloud)

---

## 🙋‍♀️ Author

**Suhwa Seong**
M.Sc. Data Science Student, UE Germany
GitHub: [https://github.com/SuhwaSeong](https://github.com/SuhwaSeong)

```
