
import argparse, os
import faiss, torch
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--curated", required=True, help="Parquet path of curated dataset")
parser.add_argument("--out", required=True, help="Local folder to store FAISS index & mapping")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

spark = SparkSession.builder.appName("BuildFAISS").getOrCreate()
df = spark.read.parquet(args.curated).select("patient_id","reason_for_visit","treatment_description")
rows = df.na.fill("").select("patient_id","reason_for_visit","treatment_description").limit(200000).collect()
spark.stop()

# Build text corpus (short summaries)
texts, ids = [], []
for r in rows:
    txt = f"patient:{r['patient_id']} reason:{r['reason_for_visit']} treatment:{r['treatment_description']}"
    texts.append(txt)
    ids.append(r['patient_id'])

# # Encoding : convert words to numeric format
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)

# Building the Search Engine : similer patient link togther.
# like sorting then do binary search fast.
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(emb.shape[1])
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(emb)
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), os.path.join(args.out,"kb.index"))

# Save id mapping
# saving the  sorted array 
with open(os.path.join(args.out,"ids.txt"),"w") as f:
    for i in ids: f.write(str(i)+"\n")

print("FAISS index built:", gpu_index.ntotal)