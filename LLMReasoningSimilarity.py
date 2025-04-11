import pandas as pd
import ollama
import json

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
                print(f"{row1['Titel']}, {row1['Beschreibung']} {row2['Titel']}, {row2['Beschreibung']} \n Bewertung: {bewertung}")

                if bewertung['Wert'] >= 0.7:
                    results.append({
                        'Titel_1': row1['Titel'],
                        'Beschreibung_1': row1['Beschreibung'],
                        'Titel_2': row2['Titel'],
                        'Beschreibung_2': row2['Beschreibung'],
                        'Wert': bewertung['Wert'],
                        'Begründung': bewertung['Begründung']
                    })
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"############################ Retry {retry_count}")
                else:
                    print(f"############################ Failed {max_retries} attempts. Skipping.")
                    continue

results_df = pd.DataFrame(results)
results_df.to_csv('reasongingSimilarityOutput.csv', index=False)