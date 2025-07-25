import pandas as pd
import re
from collections import Counter

# Define standard replacements for noisy labels
REPLACEMENTS = {
    "kleber": "klebstoff",
    "kleberreste": "klebstoff",
    "klebstoffreste": "klebstoff",
    "klebstoffe": "klebstoff",
    "klebereste": "klebstoff",
    "bitumenreste": "bitumen",
    "bitumenbahnen": "bitumen",
    "bitumendickschicht": "bitumen",
    "kunststoff/bitumen": "bitumen",
    "beschichtungen": "beschichtung",
    "reaktionsharzbeschichtung": "beschichtung",
    "beschichtet": "beschichtung",
    "dämmstoffe": "dämmstoff",
    "dämmstoffreste": "dämmstoff",
    "dämmstoffen": "dämmstoff",
    "gipsspachtel": "gips",
    "gipskarton": "gips",
    "gipsputz": "gips",
    "putze": "putz",
    "kalkmörtel": "mörtel",
    "kalkzementmörtel": "mörtel",
    "feuchteabdichtung": "abdichtung",
    "flüssigabdichtungen": "abdichtung",
    "abdichtungen": "abdichtung",
    "kunststoffen": "kunststoff",
    "bodenbelagsreste": "belagsreste",
    "klebespachtel": "klebstoff",
    "massivbaustoffen": "massivbaustoff",
    "stahlbewehrung": "bewehrung",
    "bewehrungsstahl": "bewehrung",
    "naturfarbe": "farbe"
}

STOPWORDS = {
    "ohne", "mit", "verunr", "geringf", "fremd", "/störstoffe", "verunreinigt",
    "geringfügig", "konv", "in", "z", "b", "wdvs", "geringen", "mengen"
}

def tokenize(text: str):
    return re.findall(r"[\w/]+", text.lower())

def clean_tokens(tokens):
    return [REPLACEMENTS.get(t, t) for t in tokens if t not in STOPWORDS]

def extract_top_terms(df, column, top_n=20):
    token_lists = df[column].dropna().apply(tokenize).apply(clean_tokens)
    all_tokens = [t for tokens in token_lists for t in tokens]
    top_terms = [term for term, _ in Counter(all_tokens).most_common(top_n)]
    return top_terms

def create_binary_labels(df, token_column, top_terms):
    for term in top_terms:
        df[f"label_{term}"] = df[token_column].apply(lambda toks: int(term in toks))
    return df

def preprocess_contaminant_labels(df, desc_col="Fremd-/Störstoffbeschreibung", top_n=20):
    df = df.copy()
    df = df[df[desc_col].notna() & (df[desc_col].str.lower() != "ohne fremd-/störstoffe")].copy()
    df["tokens"] = df[desc_col].apply(tokenize).apply(clean_tokens)
    top_terms = extract_top_terms(df, desc_col, top_n)
    df = create_binary_labels(df, "tokens", top_terms)
    label_cols = [f"label_{term}" for term in top_terms]
    return df, label_cols