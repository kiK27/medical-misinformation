from sentence_transformers import SentenceTransformer, util
import openai
import sqlite3
import json
import numpy as np

# Load model
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Load from DB
def load_claims_from_db(db_path='claims.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT claim, topic, source, embedding FROM claims")
    rows = c.fetchall()
    conn.close()

    claims = []
    embeddings = []
    for row in rows:
        claim, topic, source, embedding_str = row
        claims.append({"claim": claim, "topic": topic, "source": source})
        embeddings.append(np.array(json.loads(embedding_str), dtype=np.float32))

    return claims, np.vstack(embeddings)

trusted_claims, trusted_embeddings = load_claims_from_db()




import openai
openai.api_key = "add_ur_api_key"

def gpt_verify_with_claim(sentence, claim):
    prompt = f"""You are a medical fact-checking assistant.

Claim from a trusted source: "{claim}"
User-submitted sentence: "{sentence}"

Determine whether the user’s sentence is:
- Clearly supported by the trusted claim (output "verified")
- Partially related or plausible but not clearly supported (output "uncertain")
- Contradictory or potentially false (output "misinfo")

Only respond with one word: verified, uncertain, or misinfo.
"""

    print("\n[DEBUG] Prompt to GPT:\n", prompt)

    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=10,
    temperature=0
    )

    result = response.choices[0].message.content.strip().lower()


    
    print("[DEBUG] GPT response:", result)
    return result


def check_misinformation(text, lower_threshold=0.2, upper_threshold=0.68):
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    results = []
    for i, sentence in enumerate(sentences):
        sims = util.pytorch_cos_sim(sentence_embeddings[i], trusted_embeddings)[0]
        max_sim = sims.max().item()
        closest_index = sims.argmax().item()
        closest_claim = trusted_claims[closest_index] if max_sim >= lower_threshold else {"claim": "No sufficiently similar trusted claim found", "topic": "N/A", "source": "N/A"}

        if max_sim >= upper_threshold:
            status = "verified"
        elif max_sim < 0.3:
            status = "misinfo"  # or "uncertain"
        else:
            status = gpt_verify_with_claim(sentence, closest_claim["claim"])

        results.append({
            "sentence": sentence,
            "similarity": round(max_sim, 2),
            "status": status,
            "is_misinfo": status == "misinfo",
            "is_uncertain": status == "uncertain",
            "is_verified": status == "verified",
            
        })

    return results

