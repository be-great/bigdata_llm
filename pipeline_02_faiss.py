import argparse, os
import numpy as np
import torch, faiss
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

base = "data/curated_parquet"
out = "data/output/kb"
os.makedirs(out, exist_ok=True)

spark = SparkSession.builder.appName("BuildFAISS").getOrCreate()

# Load parquet files
patients  = spark.read.parquet(f"{base}/patients.parquet").na.fill("")
doctors   = spark.read.parquet(f"{base}/doctors.parquet").na.fill("")
appts     = spark.read.parquet(f"{base}/appointments.parquet").na.fill("")
treats    = spark.read.parquet(f"{base}/treatments.parquet").na.fill("")
billing   = spark.read.parquet(f"{base}/billing.parquet").na.fill("")


""""To convert text sentences into numerical embeddings"""
# Load SentenceTransformer model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
"""Function to build a knowledge base"""
# turns each DataFrame row into a text
# sentence and creates a corresponding
# unique ID list. Example : patient Ali Bob
def build_kb(df, name, text_func):
    rows = df.collect()
    texts, ids = [], []
    for r in rows:
        txt = text_func(r)
        texts.append(txt)
        ids.append(str(r[0]))  # first column = id
    # The encoding step
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)
    d = emb.shape[1]
    # Build FAISS index
    res = faiss.StandardGpuResources() if torch.cuda.is_available() else None
    index = faiss.IndexFlatL2(d)
    if res:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(emb.astype(np.float32))
    # Save FAISS index and texts
    faiss.write_index(index, f"{out}/{name}.index")
    np.save(f"{out}/{name}_facts.npy", np.array(texts))
    print(f"✅ {name.capitalize()} knowledge base saved ({len(texts)} facts)")

# ---- Define text builders for each dataset ----
build_kb(
    patients, "patients",
    lambda r: f"Patient {r['patient_first_name']} {r['patient_last_name']} ({r['gender']}) born {r['date_of_birth']} lives at {r['address']} with insurance from {r['insurance_provider']}."
)

build_kb(
    doctors, "doctors",
    lambda r: f"Doctor {r['doctor_first_name']} {r['doctor_last_name']} specializes in {r['specialization']} with {r['years_experience']} years experience at {r['hospital_branch']}."
)

build_kb(
    appts, "appointments",
    lambda r: f"Appointment {r['appointment_id']} on {r['appointment_date']} at {r['appointment_time']} for reason {r['reason_for_visit']} with doctor {r['doctor_id']} and patient {r['patient_id']}."
)

build_kb(
    treats, "treatments",
    lambda r: f"Treatment {r['treatment_id']} of type {r['treatment_type']} described as {r['description']} costing {r['cost']} on {r['treatment_date']}."
)

build_kb(
    billing, "billing",
    lambda r: f"Billing record {r['bill_id']} dated {r['bill_date']} amount {r['amount']} paid via {r['payment_method']} (status {r['payment_status']})."
)

spark.stop()
