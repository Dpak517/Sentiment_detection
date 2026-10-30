import streamlit as st
from transformers import pipeline

@st.cache_resource
def load():
    return pipeline("text-classification",
                    model="Deepakwithnotverma/bert-finetuned-sentiment", top_k=None)

clf = load()
st.title("Emotion Classifier")
text = st.text_area("Enter text", "i feel so happy today")
if st.button("Predict"):
    res = clf(text)
    scores = res[0] if isinstance(res[0], list) else res
    for s in sorted(scores, key=lambda x: -x["score"]):
        st.write(f"{s['label']}: {s['score']:.2%}")
