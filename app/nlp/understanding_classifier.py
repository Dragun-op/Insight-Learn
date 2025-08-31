import requests

CANDIDATE_LABELS = ["Understood", "Memorized", "Confused"]
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

def classify_explanation(text, hf_api_key):
    if not hf_api_key:
        raise ValueError("Missing Hugging Face API Key")

    headers = {
        "Authorization": f"Bearer {hf_api_key}"
    }

    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": CANDIDATE_LABELS}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")

    result = response.json()
    scores = {label: 0.0 for label in CANDIDATE_LABELS}
    for label, score in zip(result["labels"], result["scores"]):
        scores[label] = score

    prediction = result["labels"][0]

    return {
        "text": text,
        "prediction": prediction,
        "scores": scores
    }