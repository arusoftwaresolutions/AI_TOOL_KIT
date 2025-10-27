"""
spaCy NER and simple rule-based sentiment analysis for sample Amazon-style product reviews.
- Uses small in-script sample dataset (no uploads)
- Extracts named entities and heuristically identifies product names/brands
- Performs simple sentiment scoring using a small lexicon
Run:
    pip install -U spacy matplotlib
    python -m spacy download en_core_web_sm
    python task3_spacy_ner_sentiment.py
"""
import spacy
from spacy.matcher import Matcher
from collections import Counter
import re

# Load model (assume en_core_web_sm installed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit("Please install spaCy and the en_core_web_sm model: python -m spacy download en_core_web_sm")

# Sample Amazon-like reviews (small set embedded)
reviews = [
    "I bought the Acme Turbo Blender and it's fantastic — the motor is strong and smoothies are smooth.",
    "Terrible experience with BrandX headphones. They stopped working after a week.",
    "The PixelMax phone by NexTech is sleek; battery life is great but camera struggles in low light.",
    "Love my new CozySleep mattress (CozySleep). Best sleep I've had in years!",
    "Product: UltraSound Earbuds. Sound quality is good but ear fit is uncomfortable."
]

# Simple sentiment lexicon
positive_words = set(["good","great","fantastic","love","best","smooth","sleek","strong","excellent"])
negative_words = set(["terrible","worst","bad","stopped","uncomfortable","struggles","poor"])

def sentiment_score(text):
    # Lowercase and simple tokenization
    tokens = re.findall(r"\w+", text.lower())
    pos = sum(tok in positive_words for tok in tokens)
    neg = sum(tok in negative_words for tok in tokens)
    # Score: pos - neg
    score = pos - neg
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"score": score, "label": label, "pos": pos, "neg": neg}

# Named entity extraction and heuristic product/brand identification
def extract_entities(doc):
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    # Heuristic: collect ORG/PRODUCT/WORK_OF_ART/PERSON and proper-noun + noun phrases
    candidates = set()
    for ent in doc.ents:
        if ent.label_ in ("ORG","PRODUCT","PERSON","WORK_OF_ART"):
            candidates.add(ent.text)
    # Additional heuristic: look for patterns like "Product: NAME" or capitalized tokens
    for match in re.finditer(r"(Product|product|Product:)\s*[:\-]?\s*([A-Z][A-Za-z0-9_\-]+(?:\s[A-Z][A-Za-z0-9_\-]+)*)", doc.text):
        candidates.add(match.group(2))
    # Proper noun + noun chunks
    for chunk in doc.noun_chunks:
        if any(tok.pos_ == "PROPN" for tok in chunk):
            candidates.add(chunk.text)
    return list(candidates), ents

# Process reviews
all_entities = []
sentiments = []
print("Review Analysis")
print("---------------\n")
for i, rev in enumerate(reviews, 1):
    doc = nlp(rev)
    products, ents = extract_entities(doc)
    sent = sentiment_score(rev)
    all_entities.extend(products)
    sentiments.append(sent)
    print(f"Review {i}: {rev}")
    print("  Named Entities (spaCy):", ents)
    print("  Heuristic products/brands:", products)
    print(f"  Sentiment: {sent['label']} (score={sent['score']}, +{sent['pos']}/-{sent['neg']})\n")

# Summary
print("Summary of extracted product/brand candidates:")
print(Counter(all_entities))
print("\nAggregate sentiment counts:")
cnt = Counter([s['label'] for s in sentiments])
print(cnt)