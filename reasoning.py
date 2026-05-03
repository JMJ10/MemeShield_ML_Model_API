import re
import torch
from nltk.stem import WordNetLemmatizer
from torch.nn.functional import cosine_similarity

lemmatizer = WordNetLemmatizer()

OFFENSE_KEYWORDS = {
    "obscenity": {"sex":2.0,"porn":2.0,"fuck":1.5,"nude":1.5},
    "harassment": {"idiot":1.5,"stupid":1.5,"loser":1.2},
    "threat": {"kill":2.0,"destroy":1.5,"burn":1.5},
    "hate_speech": {"religion":1.0,"caste":1.5,"muslim":1.5,"hindu":1.5}
}


CATEGORY_PROMPTS = {
    "hate_speech": "This text expresses hate toward a group.",
    "harassment": "This text insults or mocks someone.",
    "threat": "This text expresses violence or threat.",
    "obscenity": "This text contains sexual or explicit content."
}

CYBER_LAW_MAPPING = {
    "obscenity": ["IT Act Section 67 – Publishing obscene content"],
    "harassment": ["IPC Section 504 – Intentional insult"],
    "threat": ["IPC Section 506 – Criminal intimidation"],
    "hate_speech": ["IPC Section 153A – Promoting enmity between groups"]
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

def keyword_scoring(text):
    scores = {}
    for cat, kws in OFFENSE_KEYWORDS.items():
        score = 0
        for kw, weight in kws.items():
            if re.search(rf"\b{kw}\b", text):
                score += weight
        if score > 0:
            scores[cat] = score
    return scores

def bert_scoring(text, tokenizer, bert_model, device):
    scores = {}

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        text_vec = bert_model(**inputs).pooler_output

    for cat, prompt in CATEGORY_PROMPTS.items():
        p_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            p_vec = bert_model(**p_inputs).pooler_output

        sim = cosine_similarity(text_vec, p_vec).item()

        if sim > 0.45:
            scores[cat] = sim

    return scores

def reasoning_layer(text, model_output, tokenizer, bert_model, device):

    text = normalize_text(text)

    kw_scores = keyword_scoring(text)
    bt_scores = bert_scoring(text, tokenizer, bert_model, device)

    combined = {}
    for cat in set(kw_scores) | set(bt_scores):
        combined[cat] = 0.6*kw_scores.get(cat,0) + 0.4*bt_scores.get(cat,0)

    categories = sorted(combined, key=combined.get, reverse=True)

    score = model_output["score"]

    if score > 0.85:
        conf = "high"
    elif score > 0.65:
        conf = "moderate"
    else:
        conf = "low"

    laws = []
    for c in categories:
        laws.extend(CYBER_LAW_MAPPING.get(c, []))

    laws = list(set(laws))

    if categories:
        explanation = (
            f"The model predicts this content as offensive with {conf} confidence. "
            f"Semantic analysis suggests {', '.join(categories)} patterns."
        )
    else:
        explanation = (
            f"The model predicts this content as offensive with {conf} confidence, "
            f"but no strong category was detected."
        )

    return {
        "categories": categories,
        "laws": laws,
        "explanation": explanation
    }