import argparse, os
import numpy as np
import torch, faiss
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

# Define placeholder paths directly for notebook execution
curated_path = "data/output/curated_parquet/part.parquet"
output_folder = "data/output/"

os.makedirs(output_folder, exist_ok=True)

spark = SparkSession.builder.appName("BuildFAISS").getOrCreate()
# load the same final_join parquet
df = spark.read.parquet(curated_path).na.fill("")

# select only real columns that exist in your curated parquet
df = df.select(
    "patient_id", "patient_first_name", "patient_last_name", "gender",
    "date_of_birth", "contact_number", "address", "registration_date",
    "insurance_provider", "insurance_number", "patient_email",
    "doctor_id", "doctor_first_name", "doctor_last_name", "specialization",
    "phone_number", "years_experience", "hospital_branch", "doctor_email",
    "appointment_id", "appointment_date", "appointment_time",
    "reason_for_visit", "status",
    "treatment_id", "treatment_type", "description", "cost", "treatment_date",
    "bill_id", "bill_date", "amount", "payment_method", "payment_status"
)

rows = df.limit(20000).collect()
spark.stop()
# Build text corpus (short summaries)
texts, ids = [], []
for r in rows:
    fact = (
        f"Patient {r['patient_first_name']} {r['patient_last_name']} "
        f"(ID {r['patient_id']}, {r['gender']}) lives at {r['address']} "
        f"and is insured by {r['insurance_provider']}. "
        f"They had appointment {r['appointment_id']} with Dr. "
        f"{r['doctor_first_name']} {r['doctor_last_name']} "
        f"({r['specialization']}, {r['hospital_branch']}) "
        f"on {r['appointment_date']} at {r['appointment_time']} "
        f"for {r['reason_for_visit']}. "
        f"The treatment was '{r['treatment_type']}' described as "
        f"'{r['description']}' costing {r['cost']} dollars on {r['treatment_date']}. "
        f"Bill {r['bill_id']} dated {r['bill_date']} amount {r['amount']} dollars "
        f"paid by {r['payment_method']} (status: {r['payment_status']})."
    )
    texts.append(fact)
    ids.append(f"{r['patient_id']}_{r['appointment_id']}_{r['doctor_id']}")

print(f"Generated {len(texts)} hospital knowledge sentences.")


# # Encoding : convert words to numeric format
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)
d = emb.shape[1]
# Building the Search Engine : similer patient link togther.
# like sorting then do binary search fast.
res = faiss.StandardGpuResources() if torch.cuda.is_available() else None
index = faiss.IndexFlatL2(d)
if res:
    index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(emb.astype(np.float32))

# Save id mapping
# saving the  sorted array 
faiss.write_index(index, os.path.join(output_folder, "kb.index"))
np.save(os.path.join(output_folder, "facts.npy"), np.array(texts))

print("Knowledge base saved (doctors + patients + treatments + billing).")
