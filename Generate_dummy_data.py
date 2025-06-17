import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Sample doctors and specializations
doctors = [
    ("Dr. Alice Smith", "Cardiologist"),
    ("Dr. Bob Johnson", "Dermatologist"),
    ("Dr. Carol Williams", "Endocrinologist"),
    ("Dr. David Brown", "Neurologist"),
    ("Dr. Emma Jones", "General Physician")
]

# Sample diagnoses and corresponding medicines
diagnoses_meds = {
    "Type 2 Diabetes": "Metformin",
    "Hypertension": "Lisinopril",
    "Acne": "Doxycycline",
    "Migraine": "Sumatriptan",
    "Common Cold": "Paracetamol",
    "High Cholesterol": "Atorvastatin",
    "Asthma": "Salbutamol",
    "Anxiety": "Sertraline"
}

# Generate 100 dummy records
data = []
for i in range(1, 101):
    patient_id = f"P{i:03d}"
    patient_name = fake.name()
    gender = random.choice(["Male", "Female", "Other"])
    age = random.randint(10, 80)
    dob = datetime.now() - timedelta(days=365 * age)
    consultation_id = f"C{i:03d}"
    consultation_date = fake.date_between(start_date='-1y', end_date='today')
    doctor_name, specialization = random.choice(doctors)
    diagnosis = random.choice(list(diagnoses_meds.keys()))
    prescribed_medicines = diagnoses_meds[diagnosis]
    symptoms = fake.sentence(nb_words=6)
    prescription_notes = f"Take {prescribed_medicines} as prescribed by {doctor_name}."
    consultation_notes = f"Patient reported symptoms like {symptoms}. Diagnosis was {diagnosis}. {prescription_notes}"
    follow_up_required = random.choice(["Yes", "No"])
    next_appointment_date = consultation_date + timedelta(days=random.choice([7, 14, 30])) if follow_up_required == "Yes" else ""
    urgency_level = random.choice(["Low", "Moderate", "High"])
    created_at = datetime.now()
    updated_at = datetime.now()

    data.append([
        patient_id, patient_name, gender, age, dob.date(), consultation_id,
        consultation_date, doctor_name, specialization, diagnosis, symptoms,
        prescribed_medicines, prescription_notes, consultation_notes,
        follow_up_required, next_appointment_date, urgency_level,
        created_at, updated_at
    ])

# Define column names
columns = [
    "patient_id", "patient_name", "gender", "age", "dob", "consultation_id",
    "consultation_date", "doctor_name", "specialization", "diagnosis", "symptoms",
    "prescribed_medicines", "prescription_notes", "consultation_notes",
    "follow_up_required", "next_appointment_date", "urgency_level",
    "created_at", "updated_at"
]

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=columns)
csv_path = "dummy_healthcare_data.csv"
df.to_csv(csv_path, index=False)

csv_path
