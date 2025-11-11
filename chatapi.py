import os, numpy as np, torch, faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_file = "data/output/"
# ======== CONFIG ========
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = base_file + "adapters"          # fine-tuned QLoRA adapter
FAISS_INDEX = base_file + "kb.index"
FACTS_FILE = base_file + "facts.npy"

# ======== LOAD DATA ========
print("🔹 Loading FAISS index and facts...")
index = faiss.read_index(FAISS_INDEX)
facts = np.load(FACTS_FILE, allow_pickle=True)
print(f"✅ Loaded {len(facts)} knowledge facts")

# ======== LOAD EMBEDDING MODEL ========
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ======== LOAD QLoRA / LoRA MODEL ON CPU ========
print("🔹 Loading LoRA/Glora fine-tuned model (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model in full precision on CPU
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",       # force CPU
    torch_dtype=torch.float32
)

# Apply LoRA/Glora adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("✅ LoRA/Glora model loaded successfully on CPU!")

# ======== FASTAPI SETUP ========
app = FastAPI(title="Hospital Chat (TinyLlama + QLoRA + FAISS)")

class Question(BaseModel):
    query: str
    top_k: int = 3
    max_new_tokens: int = 128

# ======== RAG RETRIEVAL FUNCTION ========
def retrieve_context(query, top_k=3):
    q_vec = embedder.encode([query])
    D, I = index.search(q_vec.astype("float32"), top_k)
    return "\n".join(facts[i] for i in I[0])

# ======== CHAT ENDPOINT ========
@app.post("/ask")
def ask(q: Question):
    context = retrieve_context(q.query, q.top_k)
    prompt = f"Context:\n{context}\n\nQuestion: {q.query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=q.max_new_tokens, temperature=0.4, top_p=0.9)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"question": q.query, "answer": answer}

# ======== LOCAL TEST ========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

