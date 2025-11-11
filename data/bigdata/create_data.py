import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Approx rows for ~10 GB total
rows_patients = 2_000_000
rows_doctors = 200_000
rows_appointments = 5_000_000
rows_treatments = 5_000_000
rows_billing = 5_000_000
chunk_size = 500_000

# ---------- Helper to write CSV in chunks ----------
def write_large_csv(filename, n_rows, generate_chunk_fn):
    for i in range(0, n_rows, chunk_size):
        chunk_rows = min(chunk_size, n_rows - i)
        df = generate_chunk_fn(chunk_rows, i)
        mode = 'w' if i == 0 else 'a'
        header = True if i == 0 else False
        df.to_csv(filename, mode=mode, index=False, header=header)
        print(f"Wrote {i+chunk_rows}/{n_rows} rows to {filename}")

# ---------- Generate Patients ----------
def gen_patients(n, offset=0):
    return pd.DataFrame({
        "patient_id": range(1+offset, 1+offset+n),
        "first_name": [fake.first_name() for _ in range(n)],
        "last_name": [fake.last_name() for _ in range(n)],
        "gender": np.random.choice(["M","F"], n),
        "date_of_birth": [fake.date_of_birth(minimum_age=0, maximum_age=90) for _ in range(n)],
        "contact_number": [fake.phone_number() for _ in range(n)],
        "address": [fake.address().replace("\n", " ") for _ in range(n)],
        "registration_date": [fake.date_between(start_date='-5y', end_date='today') for _ in range(n)],
        "insurance_provider": np.random.choice(["Aetna","BlueCross","Cigna","UnitedHealth"], n),
        "insurance_number": [fake.bothify(text='??#####') for _ in range(n)],
        "email": [fake.email() for _ in range(n)]
    })

write_large_csv("patients.csv", rows_patients, gen_patients)

# ---------- Generate Doctors ----------
specialties = ["Cardiology","Neurology","Pediatrics","Oncology","Orthopedics","Dermatology",
               "Gastroenterology","Endocrinology","Ophthalmology","Radiology"]
hospital_branches = ["North","South","East","West","Central"]

def gen_doctors(n, offset=0):
    return pd.DataFrame({
        "doctor_id": range(1+offset, 1+offset+n),
        "first_name": [fake.first_name() for _ in range(n)],
        "last_name": [fake.last_name() for _ in range(n)],
        "specialization": np.random.choice(specialties, n),
        "phone_number": [fake.phone_number() for _ in range(n)],
        "years_experience": np.random.randint(1, 40, n),
        "hospital_branch": np.random.choice(hospital_branches, n),
        "email": [fake.email() for _ in range(n)]
    })

write_large_csv("doctors.csv", rows_doctors, gen_doctors)

# ---------- Generate Appointments ----------
appointment_status = ["Scheduled","Completed","Cancelled"]
appointment_reasons = ["Checkup","Consultation","Follow-up","Emergency","Surgery"]

def gen_appointments(n, offset=0):
    return pd.DataFrame({
        "appointment_id": range(1+offset, 1+offset+n),
        "patient_id": np.random.randint(1, rows_patients+1, n),
        "doctor_id": np.random.randint(1, rows_doctors+1, n),
        "appointment_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(n)],
        "appointment_time": [fake.time() for _ in range(n)],
        "reason_for_visit": np.random.choice(appointment_reasons, n),
        "status": np.random.choice(appointment_status, n)
    })

write_large_csv("appointments.csv", rows_appointments, gen_appointments)

# ---------- Generate Treatments ----------
treatment_types = ["Checkup","Surgery","Therapy","Medication","Procedure"]

def gen_treatments(n, offset=0):
    return pd.DataFrame({
        "treatment_id": range(1+offset, 1+offset+n),
        "appointment_id": np.random.randint(1, rows_appointments+1, n),
        "treatment_type": np.random.choice(treatment_types, n),
        "description": [fake.sentence() for _ in range(n)],
        "cost": np.round(np.random.uniform(50, 5000, n),2),
        "treatment_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(n)]
    })

write_large_csv("treatments.csv", rows_treatments, gen_treatments)

# ---------- Generate Billing ----------
payment_methods = ["Credit Card","Cash","Insurance","Bank Transfer"]
payment_statuses = ["Paid","Pending","Cancelled"]

def gen_billing(n, offset=0):
    return pd.DataFrame({
        "bill_id": range(1+offset, 1+offset+n),
        "patient_id": np.random.randint(1, rows_patients+1, n),
        "treatment_id": np.random.randint(1, rows_treatments+1, n),
        "bill_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(n)],
        "amount": np.round(np.random.uniform(50, 5000, n),2),
        "payment_method": np.random.choice(payment_methods, n),
        "payment_status": np.random.choice(payment_statuses, n)
    })

write_large_csv("billing.csv", rows_billing, gen_billing)

