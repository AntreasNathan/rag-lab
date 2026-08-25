import requests, os

os.makedirs("docs", exist_ok=True)

HEADERS = {"User-Agent": "rag-learning-project/0.1 (andreas; educational use)"}

TITLES = [
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Recurrent neural network",
    "Convolutional neural network",
    "Backpropagation",
    "Gradient descent",
    "Overfitting",
    "Batch normalization",
    "Long short-term memory",
    "Generative adversarial network",
]

for title in TITLES:
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "titles": title,
                "prop": "extracts", "explaintext": 1, "redirects": 1,
            },
            headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"FAILED {title}: {e}")
        continue

    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if len(text) < 1000:
        print(f"SKIPPED {title}: only {len(text)} chars")
        continue

    fname = (title.lower().replace(" ", "_")
             .replace("(", "").replace(")", "") + ".txt")
    with open(f"docs/{fname}", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{title}: {len(text)} chars")