# create_claims_db.py
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize model
model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

# Example claims
claims_data = [
    {
        "claim": "COVID-19 vaccines are effective at reducing the risk of severe disease, hospitalization, and death from SARS-CoV-2 infection.",
        "topic": "vaccine",
        "source": "CDC"
    },
    {
        "claim": "There is no evidence that COVID-19 vaccines cause infertility or genetic changes.",
        "topic": "vaccine",
        "source": "WHO"
    },
    {
        "claim": "No dietary supplement, including turmeric or vitamin C, has been clinically proven to cure cancer.",
        "topic": "cancer",
        "source": "Mayo Clinic"
    },
    {
        "claim": "Regular cancer screenings such as mammograms or colonoscopies can significantly increase the chance of early detection and effective treatment.",
        "topic": "cancer",
        "source": "American Cancer Society"
    },
    {
        "claim": "Herbal remedies may relieve mild symptoms of stress or anxiety but are not substitutes for evidence-based medical treatment for hypertension.",
        "topic": "hypertension",
        "source": "NIH"
    },
    {
        "claim": "Controlling high blood pressure typically involves lifestyle changes like diet and exercise along with medication as needed.",
        "topic": "hypertension",
        "source": "American Heart Association"
    },
    {
        "claim": "Type 2 diabetes can be managed with lifestyle interventions, but it is not cured by diet alone in most cases.",
        "topic": "diabetes",
        "source": "CDC"
    },
    {
        "claim": "People with type 1 diabetes must use insulin therapy as their bodies do not produce insulin.",
        "topic": "diabetes",
        "source": "Mayo Clinic"
    },
    {
        "claim": "Mental illnesses are treatable medical conditions and not signs of personal weakness.",
        "topic": "mental_health",
        "source": "National Institute of Mental Health"
    },
    {
        "claim": "Treatment for depression often includes therapy, medications, or a combination, depending on severity.",
        "topic": "mental_health",
        "source": "WHO"
    },
    {
        "claim": "A heart-healthy lifestyle includes physical activity, a balanced diet, and not smoking, and can prevent many cardiovascular diseases.",
        "topic": "heart",
        "source": "American Heart Association"
    },
    {
        "claim": "Smoking greatly increases the risk of heart disease, stroke, and various cancers.",
        "topic": "heart",
        "source": "CDC"
    },
    {
        "claim": "Vaccines must meet strict safety and efficacy criteria in clinical trials before being approved for public use.",
        "topic": "vaccine",
        "source": "FDA"
    },
    {
        "claim": "Antibiotics are only effective against bacterial infections and do not treat viral illnesses like the flu or COVID-19.",
        "topic": "antibiotics",
        "source": "CDC"
    },
    {
        "claim": "Misuse of antibiotics can lead to antibiotic-resistant infections, which are harder to treat and more dangerous.",
        "topic": "antibiotics",
        "source": "WHO"
    },
    {
        "claim": "Drinking 8 glasses of water a day is not a strict medical requirement, but staying hydrated is important for overall health.",
        "topic": "hydration",
        "source": "Harvard Health"
    },
    {
        "claim": "There is no scientific evidence that detox teas or juice cleanses improve liver function or remove toxins from the body.",
        "topic": "nutrition",
        "source": "Mayo Clinic"
    },
    {
        "claim": "Pregnant women should consult their doctor before taking any supplements or herbal remedies.",
        "topic": "pregnancy",
        "source": "CDC"
    },
    {
        "claim": "Daily physical activity can reduce the risk of many chronic diseases including diabetes, heart disease, and certain cancers.",
        "topic": "exercise",
        "source": "NIH"
    },
    {
        "claim": "Washing hands with soap and water for 20 seconds is one of the most effective ways to prevent the spread of infections.",
        "topic": "infectious_disease",
        "source": "CDC"
    }
]


# Generate embeddings
for entry in claims_data:
    embedding = model.encode(entry["claim"]).tolist()
    entry["embedding"] = json.dumps(embedding)  # serialize to JSON string

# Create DB and insert
conn = sqlite3.connect('claims.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    claim TEXT,
    topic TEXT,
    source TEXT,
    embedding TEXT
)
''')

for entry in claims_data:
    c.execute('''
    INSERT INTO claims (claim, topic, source, embedding) VALUES (?, ?, ?, ?)
    ''', (entry["claim"], entry["topic"], entry["source"], entry["embedding"]))

conn.commit()
conn.close()
