# import streamlit as st
# from transformers import pipeline

# @st.cache_resource
# def load():
#     return pipeline("text-classification",
#                     model='Deepakwithnotverma/bert-finetuned-sentiment', top_k=None)

# clf = load()
# st.title("Emotion Classifier")
# text = st.text_area("Enter text")
# if st.button("Predict"):
#     res = clf(text)
#     scores = res[0] if isinstance(res[0], list) else res
#     for s in sorted(scores, key=lambda x: -x["score"]):
#         st.write(f"{s['label']}: {s['score']:.2%}")
import streamlit as st
from transformers import pipeline

REPO ='Deepakwithnotverma/bert-finetuned-sentiment'   # must match your Hub repo exactly

# Must match your training label order
NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
NAME_TO_ID = {n: i for i, n in enumerate(NAMES)}


@st.cache_resource
def load():
    return pipeline("text-classification", model=REPO, top_k=None)


def to_name_and_id(raw_label):
    """Return (label_name, label_number) for either 'joy' or 'LABEL_1'."""
    if raw_label.startswith("LABEL_"):
        idx = int(raw_label.split("_")[1])
        return NAMES[idx], idx
    return raw_label, NAME_TO_ID.get(raw_label, -1)


clf = load()

st.title("Emotion Classifier")
st.caption("Works best on short first-person sentences like 'i feel ...'")

text = st.text_area("Enter text", "i feel so happy today")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        res = clf(text.lower(), truncation=True, max_length=64)
        scores = res[0] if isinstance(res[0], list) else res
        scores = sorted(scores, key=lambda x: -x["score"])

        top_name, top_id = to_name_and_id(scores[0]["label"])
        st.success(f"Prediction: **{top_name}** (label {top_id}), "
                   f"confidence {scores[0]['score']:.1%}")

        rows = []
        for s in scores:
            name, idx = to_name_and_id(s["label"])
            rows.append({"Label number": idx, "Label name": name,
                         "Confidence": f"{s['score']:.2%}"})
        st.table(rows)
