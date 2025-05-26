import gradio as gr
import joblib
import re

# ✅ Load model and vectorizer
try:
    rf_model = joblib.load("rf_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    print("Model loading error:", e)
    rf_model = None
    tfidf = None

# ✅ Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ✅ Prediction function
def predict_news(text):
    if not text.strip():
        return "❌ Please enter valid news text."
    if rf_model is None or tfidf is None:
        return "🚫 Model not loaded. Check rf_model.pkl and tfidf_vectorizer.pkl files."
    try:
        cleaned = clean_text(text)
        vect = tfidf.transform([cleaned])
        prediction = rf_model.predict(vect)
        return f"✅ Prediction: {prediction[0]}"
    except Exception as e:
        return f"🚫 Error during prediction: {str(e)}"

# ✅ Gradio interface
iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Paste news article here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector",
    description="Paste any news article below to check whether it's Fake or Real."
)

if __name__ == "__main__":
    iface.launch()
