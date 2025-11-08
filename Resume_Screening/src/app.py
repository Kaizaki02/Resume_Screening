# # app.py (core parts)
# import streamlit as st
# import pickle, re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# import fitz  # PyMuPDF
# import nltk
# nltk.download('punkt'); nltk.download('stopwords')

# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# def clean_resume(text):
#     text = str(text).lower()
#     text = re.sub(r'[^a-z\s]', ' ', text)
#     tokens = word_tokenize(text)
#     tokens = [ps.stem(w) for w in tokens if w not in stop_words]
#     return ' '.join(tokens)

# # Load artifacts
# tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
# le = pickle.load(open("label_encoder.pkl", "rb"))
# svc_model = pickle.load(open("model_svm.pkl", "rb"))

# def extract_text_from_pdf(uploaded_file):
#     text = ""
#     with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
#         for page in doc:
#             text += page.get_text()
#     return text

# def predict_topk(text, k=3):
#     cleaned = clean_resume(text)
#     vec = tfidf.transform([cleaned])
#     pred = svc_model.predict(vec)
#     label = le.inverse_transform(pred)[0]
#     topk = []
#     if hasattr(svc_model, "predict_proba"):
#         probs = svc_model.predict_proba(vec)[0]
#         idx = probs.argsort()[-k:][::-1]
#         topk = [(le.inverse_transform([i])[0], round(probs[i]*100,2)) for i in idx]
#     else:
#         topk = [(label, 100.0)]
#     return label, topk

# st.title("Resume Screening App")
# option = st.radio("Input type", ["Paste Resume Text", "Upload PDF"])
# resume_text = ""
# if option == "Paste Resume Text":
#     resume_text = st.text_area("Paste resume here", height=250)
# else:
#     uploaded = st.file_uploader("Upload PDF", type=["pdf"])
#     if uploaded:
#         resume_text = extract_text_from_pdf(uploaded)
#         st.success("Text extracted")

# if st.button("Predict"):
#     if not resume_text.strip():
#         st.warning("Provide resume text or file.")
#     else:
#         label, top3 = predict_topk(resume_text)
#         st.success(f"Predicted Category: {label}")
#         st.subheader("Top matches")
#         for i,(lab,score) in enumerate(top3,1):
#             st.write(f"{i}. {lab} — {score}%")


# # inside your app or separate script
# from sklearn.metrics.pairwise import cosine_similarity
# clean_jd = clean_resume(job_description_text)
# jd_vec = tfidf.transform([clean_jd])
# resume_vecs = tfidf.transform(df['Cleaned_Resume'].tolist())
# scores = cosine_similarity(jd_vec, resume_vecs).flatten()
# df['MatchScore'] = scores * 100
# df_sorted = df.sort_values('MatchScore', ascending=False)
