from langchain_ollama import OllamaEmbeddings
from numpy import dot
from numpy.linalg import norm
import math

def cosine_similarity(v1, v2):
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(a * a for a in v2))
    
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Example sentences to compare
sentence1 = "The quick brown fox jumps over the lazy dog"
sentence2 = "A fast brown dog leaps above a sleepy canine"

# Get embeddings for both sentences
embedding1 = embeddings.embed_query(sentence1)
embedding2 = embeddings.embed_query(sentence2)

# Calculate similarity score
similarity = cosine_similarity(embedding1, embedding2)

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Similarity score: {similarity:.4f}")
