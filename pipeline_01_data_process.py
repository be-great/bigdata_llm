import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim

parser = argparse.ArgumentParser()
parser.add_argument("--inbase", required=True, help="Base dir containing data_csv (local or hdfs://)")
parser.add_argument("--out", required=True, help="Output Parquet folder (local or hdfs:///)")
args = parser.parse_args()

spark = SparkSession.builder.appName("HospitalCurate").getOrCreate()

def read_csv(name):
    return (spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(f"{args.inbase}/data_csv/{name}.csv"))

patients     = read_csv("patients")
doctors      = read_csv("doctors")
appointments = read_csv("appointments")
treatments   = read_csv("treatments")
billing      = read_csv("billing")

def clean(df):
    # lower+trim only for string columns
    return df.select([lower(trim(c)).alias(c) if t.simpleString()=="string" else col(c)
                      for c,t in zip(df.columns,[f.dataType for f in df.schema.fields])])

patients_c   = clean(patients)
doctors_c    = clean(doctors)
appts_c      = clean(appointments)
treats_c     = clean(treatments)
bill_c       = clean(billing)


# Rename duplicate columns before join
patients_c = patients_c.withColumnRenamed("email", "patient_email")
doctors_c  = doctors_c.withColumnRenamed("email", "doctor_email")
patients_c = patients_c.withColumnRenamed("first_name", "patient_first_name")
patients_c = patients_c.withColumnRenamed("last_name", "patient_last_name")
doctors_c  = doctors_c.withColumnRenamed("first_name", "doctor_first_name")
doctors_c  = doctors_c.withColumnRenamed("last_name", "doctor_last_name")

# joins
ap_pat        = appts_c.join(patients_c, "patient_id", "inner")
ap_pat_doc    = ap_pat.join(doctors_c, "doctor_id", "inner")
ap_pat_doc_tr = ap_pat_doc.join(treats_c, "appointment_id", "inner")
final_join    = ap_pat_doc_tr.join(bill_c, ["patient_id","treatment_id"], "inner").dropDuplicates()

# write curated Parquet
final_join.write.mode("overwrite").parquet(f"{args.out}/curated_parquet")

# # optional summaries for your 5 questions
# from pyspark.sql.functions import count, avg, month, to_date
# summary_by_doctor = (final_join.groupBy("doctor_id")
#                      .agg(count("*").alias("n_cases"), avg("cost").alias("avg_cost")))
# summary_by_doctor.write.mode("overwrite").parquet(f"{args.out}/summary_by_doctor")

# final_join.withColumn("month", month(to_date(col("appointment_date"))))\
#           .groupBy("month").count()\
#           .write.mode("overwrite").parquet(f"{args.out}/appts_by_month")

spark.stop()
