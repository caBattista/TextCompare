from langchain_ollama import OllamaEmbeddings
import pandas as pd
import ollama
import json
import math

embeddings = OllamaEmbeddings(model="nomic-embed-text")
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(a * a for a in v2))
    return dot_product / (magnitude1 * magnitude2)

def doText(promt):
    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user","content": promt}],
    )
    return response['message']['content'].strip()


df1 = pd.read_csv("input1.csv")
df2 = pd.read_csv("input2.csv")

results = []
for _1, row1 in df2.iterrows():
    for _2, row2 in df1.iterrows():
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                bewertung = doText(f"""
                Du bekommst zwei Einträge. Es geht um Risiken in einem Unternehmen. 
                Gib einen einen Wert zwischen 0 und 1 zurück, ob die beiden Einträge von der grundlegenden Bedeutung her ähnlich (außer dass sie Risiken in einem Unternehemen sind) sind mit der dazugehörigen Begründung. 
                Eintrag 1: Titel: {row1['Titel']} Beschreibung: {row1['Beschreibung']}
                Eintrag 2: Titel: {row2['Titel']} Beschreibung: {row2['Beschreibung']}
                WICHTIG! Gib ein JSON zurück NUR In folgendem Format: 
                <format>
                {{ 
                    "Wert": <Zahl zwischen 0 und 1 mit . als Dezimaltrennzeichen>,
                    "Begründung": "<Kurze Begründung>"
                }}
                </format>
                """)

                bewertung = json.loads(bewertung[bewertung.find('{'):bewertung.rfind('}')+1])
                print(f"{bewertung}")

                embedding1 = embeddings.embed_query(row1['Titel'] + " " + row1['Beschreibung'])
                embedding2 = embeddings.embed_query(row2['Titel'] + " " + row2['Beschreibung'])
                semantic_similarity = cosine_similarity(embedding1, embedding2)

                result = {
                    'Titel_1': row1['Titel'],
                    'Beschreibung_1': row1['Beschreibung'],
                    'Titel_2': row2['Titel'],
                    'Beschreibung_2': row2['Beschreibung'],
                    'Reasoning_Wert': bewertung['Wert'],
                    'Reasoning_Begründung': bewertung['Begründung'],
                    'Semantischer_Wert': semantic_similarity,
                    'Gesamtwert': (bewertung['Wert'] + semantic_similarity) / 2
                }
                print(json.dumps(result, ensure_ascii=False, indent=2))
                results.append(result)
                
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"############################ Retry {retry_count}")
                else:
                    print(f"############################ Failed {max_retries} attempts. Skipping.")
                    continue

results_df = pd.DataFrame(results)
results_df.to_csv('semanticAndReasoningSimilarityOutput.csv', index=False)