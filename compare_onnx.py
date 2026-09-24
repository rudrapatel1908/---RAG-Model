import onnxruntime as ort
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

MODEL_DIR = Path("backend/models/e5-small-onnx")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

with open("retrieval/_debug_query.txt", "r", encoding="utf-8-sig") as f:
    query_text = "query: " + f.read().strip()

passage_text = "passage: द मैनहट्टन प्रोजेक्ट। इस एक बार वर्गीकृत तस्वीर में पहला परमाणु बम - एक हथियार जिसे परमाणु वैज्ञानिकों ने गैजेट उपनाम दिया।"

def embed(sess, text):
    enc = tok(text, return_tensors="np", padding=True, truncation=True)
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = np.zeros_like(enc["input_ids"])
    input_names = [i.name for i in sess.get_inputs()]
    inputs = {k: v for k, v in enc.items() if k in input_names}
    out = sess.run(None, inputs)[0]
    emb = out[0].mean(axis=0)
    return emb / np.linalg.norm(emb)

def cos_sim(a, b):
    return float(np.dot(a, b))

for name in ["model.onnx", "model_quantized.onnx"]:
    sess = ort.InferenceSession(str(MODEL_DIR / name))
    q_emb = embed(sess, query_text)
    p_emb = embed(sess, passage_text)
    print(f"{name}: query-passage cosine sim = {cos_sim(q_emb, p_emb):.4f}")

sess_fp32 = ort.InferenceSession(str(MODEL_DIR / "model.onnx"))
sess_int8 = ort.InferenceSession(str(MODEL_DIR / "model_quantized.onnx"))
q_fp32 = embed(sess_fp32, query_text)
q_int8 = embed(sess_int8, query_text)
print(f"fp32-vs-int8 query embedding agreement (cosine) = {cos_sim(q_fp32, q_int8):.4f}")
