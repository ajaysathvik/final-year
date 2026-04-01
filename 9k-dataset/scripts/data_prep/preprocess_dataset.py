import argparse
import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = "/home/norm/Projects/Reddit-dataset/9k-dataset/Data labeling/outputs/annotations_20260401_115026.csv"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "data_preprocessed.csv"
SUMMARY_FILE = OUTPUT_DIR / "preprocessing_summary.json"

REAL_COLUMNS = [
    "annotation.is_fraud",
    "annotation.fraud_type",
    "annotation.fraud_labels.transaction_upi_fraud",
    "annotation.fraud_labels.transaction_card_fraud",
    "annotation.fraud_labels.transaction_bank_transfer",
    "annotation.fraud_labels.commerce_nondelivery",
    "annotation.fraud_labels.commerce_fake_seller",
    "annotation.fraud_labels.credential_phishing",
    "annotation.fraud_labels.social_authority_scam",
    "annotation.fraud_labels.social_urgency_scam",
    "annotation.fraud_labels.meta_victim_story",
    "annotation.fraud_labels.meta_fraud_question",
    "annotation.key_features.payment_method",
    "annotation.key_features.fraud_channel",
    "annotation.key_features.victim_action",
    "annotation.key_features.request_type",
    "annotation.key_features.impersonated_entity",
    "annotation.key_features.amount_mentioned",
    "annotation.key_features.currency",
    "annotation.key_features.urgency_level",
    "annotation.psychological_tactics.urgency",
    "annotation.psychological_tactics.fear",
    "annotation.psychological_tactics.authority",
    "annotation.psychological_tactics.reward",
]

FRAUD_TYPE_MAP = {
    "transaction": "transaction",
    "commerce": "commerce",
    "social engineering": "social_engineering",
    "social authority scam": "social_engineering",
    "phishing": "credential",
    "credential": "credential",
    "credential phishing": "credential",
    "commerce nondelivery": "commerce",
    "commerce fake seller": "commerce",
    "identity theft": "identity_theft",
    "pig butchering": "pig_butchering",
    "pig butchering scam": "pig_butchering",
    "none": "none",
    "meta": "meta",
}

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
}

CURRENCY_ALIASES = {
    "usd": "USD",
    "cad": "CAD",
    "eur": "EUR",
    "euro": "EUR",
    "gbp": "GBP",
    "inr": "INR",
    "rs": "INR",
    "btc": "BTC",
    "bitcoin": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
    "usdt": "USDT",
    "usdc": "USDC",
    "sol": "SOL",
    "tether": "USDT",
    "aud": "AUD",
    "php": "PHP",
    "mxn": "MXN",
    "pkr": "PKR",
    "qar": "QAR",
    "aed": "AED",
    "xrp": "XRP",
    "ton": "TON",
}

MULTI_VALUE_COLUMNS = {
    "annotation.key_features.payment_method",
    "annotation.key_features.fraud_channel",
    "annotation.key_features.victim_action",
    "annotation.key_features.request_type",
}

UNKNOWN_LIKE = {"", "nan", "none", "unknown", "null", "n/a", "na"}


def clean_text(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_token(token):
    token = re.sub(r"[_\-]+", " ", token.strip().lower())
    token = re.sub(r"\s+", " ", token)
    return token


def normalize_pipe_list(value):
    text = clean_text(value)
    if pd.isna(text):
        return "unknown"

    tokens = [normalize_token(part) for part in text.split("|")]
    tokens = [token for token in tokens if token]
    if not tokens:
        return "unknown"

    deduped = []
    seen = set()
    for token in tokens:
        if token == "whatsapp":
            token = "whatsapp"
        elif token == "dm":
            token = "direct_message"
        elif token == "social media":
            token = "social_media"
        elif token == "in person":
            token = "in_person"
        elif token == "door to door":
            token = "door_to_door"
        elif token == "gift card":
            token = "gift_card"
        elif token == "seed phrase":
            token = "seed_phrase"
        elif token == "shared credentials":
            token = "shared_credentials"
        elif token == "clicked link":
            token = "clicked_link"
        elif token == "installed app":
            token = "installed_app"
        elif token == "sent money":
            token = "sent_money"

        token = token.replace(" ", "_")
        if token not in seen:
            seen.add(token)
            deduped.append(token)

    if not deduped:
        return "unknown"
    return " | ".join(deduped)


def normalize_fraud_type(value):
    text = clean_text(value)
    if pd.isna(text):
        return "none"

    text = normalize_token(text)
    if "|" in text:
        tokens = [normalize_token(part) for part in text.split("|")]
        tokens = [FRAUD_TYPE_MAP.get(token, token.replace(" ", "_")) for token in tokens if token]
        tokens = [token for token in tokens if token != "none"]
        return tokens[0] if tokens else "none"

    return FRAUD_TYPE_MAP.get(text, text.replace(" ", "_"))


def normalize_currency(raw_currency, raw_amount):
    value = clean_text(raw_currency)
    if not pd.isna(value):
        normalized = normalize_token(value)
        normalized = normalized.replace(" via trc 20 network", "")
        normalized = normalized.replace("($)", "")
        normalized = normalized.strip()
        if normalized not in UNKNOWN_LIKE:
            return CURRENCY_ALIASES.get(normalized, normalized.upper().replace(" ", "_"))

    amount_text = clean_text(raw_amount)
    if pd.isna(amount_text):
        return "unknown"

    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in amount_text:
            return code

    lower_amount = amount_text.lower()
    for alias, code in CURRENCY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower_amount):
            return code

    return "unknown"


def parse_amount(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA

    lower = text.lower().replace(",", "").strip()
    if lower in UNKNOWN_LIKE:
        return pd.NA

    multiplier = 1.0
    if "lakh" in lower:
        multiplier = 100000.0
        lower = lower.replace("lakhs", "").replace("lakh", "").strip()
    elif "million" in lower:
        multiplier = 1000000.0
        lower = lower.replace("million", "").strip()
    elif re.search(r"\bmn\b", lower):
        multiplier = 1000000.0
        lower = re.sub(r"\bmn\b", "", lower).strip()
    elif re.search(r"\bk\b", lower):
        multiplier = 1000.0
        lower = re.sub(r"\bk\b", "", lower).strip()

    lower = re.sub(r"[^\d.\-]", " ", lower)
    match = re.search(r"-?\d+(?:\.\d+)?", lower)
    if not match:
        return pd.NA

    try:
        return float(match.group()) * multiplier
    except ValueError:
        return pd.NA


def normalize_binary(series, default=0):
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.fillna(default)
    return numeric.round().clip(lower=0, upper=1).astype("Int64")


def normalize_score(series, default=0.0):
    numeric = pd.to_numeric(series, errors="coerce").fillna(default)
    return numeric.clip(lower=0.0, upper=1.0)


def normalize_is_fraud(series):
    numeric = pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)
    return numeric.where(numeric.isin([-1, 0, 1]), -1)


def preprocess(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    summary_file: Path = SUMMARY_FILE,
    keep_post_id: bool = False,
):
    df = pd.read_csv(input_file)
    raw_shape = df.shape
    post_id_series = None

    if keep_post_id and "post_metadata.post_id" in df.columns:
        post_id_series = df["post_metadata.post_id"].copy()

    if set(REAL_COLUMNS).issubset(df.columns):
        df = df.copy()
    else:
        df = df.iloc[:, : len(REAL_COLUMNS)].copy()
        df.columns = REAL_COLUMNS

    for column in REAL_COLUMNS:
        if df[column].dtype == object:
            df[column] = df[column].map(clean_text)

    df["annotation.is_fraud"] = normalize_is_fraud(df["annotation.is_fraud"])
    df["annotation.fraud_type"] = df["annotation.fraud_type"].map(normalize_fraud_type)

    fraud_label_columns = [col for col in df.columns if col.startswith("annotation.fraud_labels.")]
    for column in fraud_label_columns:
        df[column] = normalize_binary(df[column], default=0)

    for column in [
        "annotation.psychological_tactics.urgency",
        "annotation.psychological_tactics.fear",
        "annotation.psychological_tactics.authority",
        "annotation.psychological_tactics.reward",
    ]:
        df[column] = normalize_score(df[column], default=0.0)

    df["annotation.key_features.urgency_level"] = normalize_score(
        df["annotation.key_features.urgency_level"], default=0.0
    )

    for column in MULTI_VALUE_COLUMNS:
        df[column] = df[column].map(normalize_pipe_list)

    df["annotation.key_features.impersonated_entity"] = (
        df["annotation.key_features.impersonated_entity"]
        .fillna("unknown")
        .map(lambda value: normalize_token(value).replace(" ", "_") if value not in {pd.NA} else "unknown")
    )

    df["annotation.key_features.amount_mentioned"] = df["annotation.key_features.amount_mentioned"].fillna("unknown")
    df["annotation.key_features.currency"] = [
        normalize_currency(currency, amount)
        for currency, amount in zip(
            df["annotation.key_features.currency"],
            df["annotation.key_features.amount_mentioned"],
        )
    ]
    df["annotation.key_features.amount_normalized"] = df["annotation.key_features.amount_mentioned"].map(parse_amount)
    df["annotation.key_features.has_amount"] = df["annotation.key_features.amount_normalized"].notna().astype("Int64")

    if post_id_series is not None:
        normalized_post_ids = post_id_series.fillna("").astype(str).str.strip()
        if "post_metadata.post_id" in df.columns:
            df["post_metadata.post_id"] = normalized_post_ids
        else:
            df.insert(0, "post_metadata.post_id", normalized_post_ids)

    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "raw_shape": list(raw_shape),
        "processed_shape": list(df.shape),
        "dropped_trailing_columns": max(raw_shape[1] - len(df.columns), 0),
        "keep_post_id": keep_post_id,
        "is_fraud_distribution": df["annotation.is_fraud"].value_counts(dropna=False).sort_index().to_dict(),
        "fraud_type_top_10": df["annotation.fraud_type"].value_counts().head(10).to_dict(),
        "currencies_top_10": df["annotation.key_features.currency"].value_counts().head(10).to_dict(),
        "parsed_amount_count": int(df["annotation.key_features.amount_normalized"].notna().sum()),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess labeled fraud annotation CSV data.")
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--summary-file", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--keep-post-id", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_file=args.input_file,
        output_file=args.output_file,
        summary_file=args.summary_file,
        keep_post_id=args.keep_post_id,
    )
