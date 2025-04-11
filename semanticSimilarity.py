from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import pandas as pd

df1 = pd.read_csv("input1.csv")
df2 = pd.read_csv("input2.csv")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = InMemoryVectorStore.from_texts(
    (df1['Titel'] + " ### " + df1['Beschreibung']).tolist(),
    embedding=embeddings
)

results = []
for _, row in df2.iterrows():
    retrieved_documents = vectorstore.similarity_search_with_score(f"{row['Titel']} {row['Beschreibung']}", k=10)
    for doc, score in retrieved_documents:
        if score >= 0.7:
            orig_titel, orig_beschreibung = doc.page_content.split(" ### ")
            results.append({
                'Titel_1': row['Titel'],
                'Beschreibung_1': row['Beschreibung'],
                'Titel_2': orig_titel,
                'Beschreibung_2': orig_beschreibung,
                'Wert': f"{score:.4f}",
            })
results_df = pd.DataFrame(results)
results_df.to_csv('semanticSimilarityOutput.csv', index=False)