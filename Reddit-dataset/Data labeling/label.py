"""
Post-Only Reddit Fraud Annotation System
----------------------------------------

Annotates ONLY Reddit posts.
Comments are used as contextual evidence.

Author: Team D16
"""

import os
import time
import json
import pandas as pd
import re
import argparse
from datetime import datetime
import requests

# ======================================================
# CONFIGURATION
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSTS_FILE = os.path.join(BASE_DIR, "posts.csv")
COMMENTS_FILE = os.path.join(BASE_DIR, "comments.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b-instruct-q5_K_M"


REQUESTS_PER_MIN = 25
MAX_TOKENS = 4096
TEMPERATURE = 0
MAX_RETRIES = 2

CACHE_VERSION = "v4_currency_fix"
CACHE_FILE = os.path.join(BASE_DIR, "post_annotation_cache.json")
LOCK_FILE = os.path.join(BASE_DIR, "cache.lock")
FAILED_POSTS_FILE = os.path.join(BASE_DIR, "failed_posts.txt")
DEBUG_ERROR_DUMP_FILE = os.path.join(BASE_DIR, "debug_failed_response.txt")



# ======================================================
# RATE LIMITER
# ======================================================

SECONDS_PER_REQ = 60 / REQUESTS_PER_MIN
_last_call_time = 0

def throttle():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < SECONDS_PER_REQ:
        time.sleep(SECONDS_PER_REQ - elapsed)
    _last_call_time = time.time()

# ======================================================
# CACHE
# ======================================================

def is_valid_annotation(obj, expected_post_id):
    if not isinstance(obj, dict):
        return False
    post_meta = obj.get("post_metadata")
    annotation = obj.get("annotation")
    if not isinstance(post_meta, dict) or not isinstance(annotation, dict):
        return False
    if str(post_meta.get("post_id", "")) != str(expected_post_id):
        return False
    if "is_fraud" not in annotation:
        return False
    return True

def normalize_annotation(obj, expected_post_id, post_row, num_comments):
    if not isinstance(obj, dict):
        return None

    post_meta = obj.get("post_metadata")
    if not isinstance(post_meta, dict):
        post_meta = {}

    post_meta["post_id"] = str(expected_post_id)
    post_meta.setdefault("subreddit", str(post_row.get("subreddit", "")))
    post_meta.setdefault("title", str(post_row.get("title", "")))
    post_meta.setdefault("body", str(post_row.get("body_text", "")))
    post_meta.setdefault("num_comments", int(num_comments))

    obj["post_metadata"] = post_meta

    annotation = obj.get("annotation")
    if not isinstance(annotation, dict) or "is_fraud" not in annotation:
        return None

    obj["annotation"] = annotation
    return obj

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[WARN] Cache JSON invalid: {exc}. Ignoring cache and re-running posts.")
        return {}

    if not isinstance(raw_cache, dict):
        print("[WARN] Cache JSON is not an object. Ignoring cache and re-running posts.")
        return {}

    cleaned = {}
    dropped = 0
    for key, value in raw_cache.items():
        if ":" in key:
            _, post_id = key.split(":", 1)
        else:
            post_id = key

        if is_valid_annotation(value, post_id):
            cleaned[key] = value
        else:
            dropped += 1

    if dropped:
        print(f"[WARN] Dropped {dropped} invalid cache entries; they will be re-annotated.")

    return cleaned

def acquire_lock():
    while os.path.exists(LOCK_FILE):
        time.sleep(0.1)
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def sync_cache():
    global CACHE
    new_data = load_cache()
    CACHE.update(new_data)

def save_cache(new_entry_key=None, new_entry_val=None):
    acquire_lock()
    try:
        # Load latest from disk
        current_disk_cache = load_cache()
        # Merge our new entry if provided
        if new_entry_key and new_entry_val:
            current_disk_cache[new_entry_key] = new_entry_val
        
        # Save back to disk
        tmp_path = f"{CACHE_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(current_disk_cache, f, indent=2)
        os.replace(tmp_path, CACHE_FILE)
        
        # Update our in-memory cache
        global CACHE
        CACHE = current_disk_cache
    finally:
        release_lock()

CACHE = load_cache()

# ======================================================
# SAFE JSON PARSER
# ======================================================

def safe_json_load(text):
    if not text or not text.strip():
        raise ValueError("Empty model response")

    # Attempt 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract substring from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1:
        # If valid end not found, or end is before start (impossible if find works), try until end of string
        if end == -1 or end < start:
            candidate = text[start:]
        else:
            candidate = text[start:end+1]
            
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt 3: Append closing brace if it looks truncated
            try:
                return json.loads(candidate + "}")
            except json.JSONDecodeError:
                pass
                
    raise ValueError("No JSON found in response")

def append_error_dump(post_id, attempt, stage, detail, raw_text=None):
    timestamp = datetime.now().isoformat()
    with open(DEBUG_ERROR_DUMP_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n---\n")
        f.write(f"time={timestamp}\n")
        f.write(f"post_id={post_id}\n")
        f.write(f"attempt={attempt}\n")
        f.write(f"stage={stage}\n")
        f.write(f"detail={detail}\n")
        if raw_text is not None:
            f.write("raw=\n")
            f.write(str(raw_text))
            f.write("\n")

# ======================================================
# COMMENT CONTEXT SUMMARY
# ======================================================

def get_comments_text(comments_df, post_id, max_comments=20):
    subset = comments_df[comments_df["post_id"] == post_id].head(max_comments)

    comments_list = []

    for _, row in subset.iterrows():
        comments_list.append(str(row.get("comment_text", ""))[:500])

    return {
        "num_comments": len(subset),
        "comments": comments_list
    }

# ======================================================
# PROMPT
# ======================================================

ANNOTATION_PROMPT = """
You are an expert payment fraud analyst.

TASK:
- Analyze the REDDIT POST only.
- Comments are provided as CONTEXT to support your decision.
- Do NOT label comments as separate events.
- Be conservative. If evidence is insufficient, set is_fraud = -1.
- Specify currency ONLY if explicitly mentioned in the text (e.g. $, €, ₹). Otherwise, use "unknown".

IMPORTANT:
- Return ONLY a valid JSON object.
- No explanations, no markdown, no extra text.

POST INFORMATION:
Subreddit: <<SUBREDDIT>>
Title: <<TITLE>>
Body: <<BODY>>

COMMENTS CONTEXT:
<<COMMENTS>>

OUTPUT SCHEMA (MUST MATCH EXACTLY):

{
  "post_metadata": {
    "post_id": "<<POST_ID>>",
    "subreddit": "string",
    "title": "string",
    "body": "string",
    "num_comments": number
  },

  "annotation": {
    "is_fraud": -1 | 0 | 1,
    "fraud_confidence": 0.0 to 1.0,
    "fraud_type": "none | transaction | commerce | credential | social_engineering | advanced | meta",

    "fraud_labels": {
      "transaction_upi_fraud": 0 or 1,
      "transaction_card_fraud": 0 or 1,
      "transaction_bank_transfer": 0 or 1,
      "commerce_nondelivery": 0 or 1,
      "commerce_fake_seller": 0 or 1,
      "credential_phishing": 0 or 1,
      "social_authority_scam": 0 or 1,
      "social_urgency_scam": 0 or 1,
      "meta_victim_story": 0 or 1,
      "meta_fraud_question": 0 or 1
    },

    "key_features": {
      "payment_method": "upi | card | bank | crypto | gift_card | unknown | none",
      "fraud_channel": "sms | email | phone | social_media | website | app | unknown | none",
      "victim_action": "sent_money | shared_credentials | clicked_link | installed_app | none | unknown",
      "request_type": "payment | verification | login | download | none | unknown",
      "impersonated_entity": "bank | government | company | individual | none | unknown",
      "amount_mentioned": "number or null",
      "currency": "USD | EUR | GBP | INR | other | unknown",
      "urgency_level": 0.0 to 1.0
    },

    "psychological_tactics": {
      "urgency": 0 or 1,
      "fear": 0 or 1,
      "authority": 0 or 1,
      "reward": 0 or 1
    },

    "community_signals": {
      "num_comments": number,
      "scam_confirmations": number,
      "not_scam_claims": number,
      "advice_requests": number
    },

    "label_quality": {
      "confidence_bucket": "low | medium | high",
      "usable_for_training": true or false
    },

    "gan_quality": {
      "suitable_for_gan": true or false,
      "quality_score": 0.0 to 1.0
    },

    "reasoning": {
      "primary_evidence": "max 100 chars",
      "uncertainty_notes": "max 100 chars or null"
    }
  }
}

DECISION RULES:
- is_fraud = 1 ONLY if actual fraud is described or clearly occurred.
- Meta content (questions, warnings, victim stories) alone is NOT fraud.
- If the post only asks whether something is a scam → is_fraud = 0.
- If evidence is weak, conflicting, or incomplete → is_fraud = -1.
- If fraud_type = meta → suitable_for_gan MUST be false.
- usable_for_training = true ONLY if confidence_bucket = high AND is_fraud ≠ -1.

"""

# ======================================================
# ANNOTATION
# ======================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Replace all quotes with single quotes to avoid JSON injection issues in the prompt
    text = text.replace('“', "'").replace('”', "'").replace('‘', "'").replace('’', "'").replace('"', "'")
    # Remove newlines and tabs from titles/subreddits to keep the schema injection clean
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text.strip()

def annotate_post(post_row, comments_df):

    post_id = str(post_row["post_id"])
    cache_key = f"{CACHE_VERSION}:{post_id}"

    # Sync with disk to see if another process finished this post
    sync_cache()
    if cache_key in CACHE:
        print(f"[CACHE HIT] {post_id}")
        return CACHE[cache_key]

    context = get_comments_text(comments_df, post_id, max_comments=5)

    prompt = ANNOTATION_PROMPT
    prompt = prompt.replace("<<POST_ID>>", post_id)
    prompt = prompt.replace("<<SUBREDDIT>>", clean_text(str(post_row.get("subreddit", ""))))
    prompt = prompt.replace("<<TITLE>>", clean_text(str(post_row.get("title", "")))[:400])
    prompt = prompt.replace("<<BODY>>", clean_text(str(post_row.get("body_text", "")))[:1500])
    
    comments_text = ""
    for i, c in enumerate(context["comments"]):
        comments_text += f"Comment {i+1}: {c}\n"
    prompt = prompt.replace("<<COMMENTS>>", comments_text)

    throttle()

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.8,
            "repeat_penalty": 1.1,
            "num_predict": MAX_TOKENS,
            "num_ctx": 4096
        }
    }

    annotation = None
    for attempt in range(MAX_RETRIES + 1):
        response = requests.post(OLLAMA_URL, json=payload)
        try:
            raw = response.json()["response"].strip()
        except Exception as exc:
            append_error_dump(post_id, attempt + 1, "response_parse", f"{type(exc).__name__}: {exc}", response.text)
            print(f"[WARN] Invalid response for post {post_id} (attempt {attempt + 1}). Saved to debug_failed_response.txt")
            continue

        try:
            parsed = safe_json_load(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            append_error_dump(post_id, attempt + 1, "json_parse", f"{type(exc).__name__}: {exc}", raw)
            print(f"[WARN] Invalid JSON for post {post_id} (attempt {attempt + 1}): {exc}. Saved to debug_failed_response.txt")
            continue

        normalized = normalize_annotation(parsed, post_id, post_row, context["num_comments"])
        if not normalized or not is_valid_annotation(normalized, post_id):
            append_error_dump(post_id, attempt + 1, "schema", "Invalid schema after normalize", raw)
            print(f"[WARN] Invalid schema for post {post_id} (attempt {attempt + 1}).")
            continue

        annotation = normalized
        break

    if annotation is None:
        raise ValueError(f"Failed to get valid JSON for post {post_id} after retries")

    # Save only this specific new annotation to the shared cache
    save_cache(cache_key, annotation)

    return annotation

# ======================================================
# RUN
# ======================================================

def run_annotation(limit=None, posts_file=POSTS_FILE, comments_file=COMMENTS_FILE, output_csv_path=None, output_json_path=None):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    posts_df = pd.read_csv(posts_file)
    comments_df = pd.read_csv(comments_file)

    # Normalize columns
    posts_df.columns = posts_df.columns.str.lower().str.strip()
    comments_df.columns = comments_df.columns.str.lower().str.strip()

    posts_df["post_id"] = posts_df["post_id"].astype(str)
    comments_df["post_id"] = comments_df["post_id"].astype(str)

    if limit:
        posts_df = posts_df.head(limit)
    else:
        # Random shuffle to minimize collisions between parallel workers
        posts_df = posts_df.sample(frac=1).reset_index(drop=True)

    results = []

    for idx, row in posts_df.iterrows():
        post_id = row['post_id']
        print(f"Annotating {idx+1}/{len(posts_df)} - {post_id}")
        try:
            ann = annotate_post(row, comments_df)
            results.append(ann)
        except Exception as e:
            print(f"[FATAL ERROR] Post {post_id} failed: {e}. Skipping and logging to failed_posts.txt")
            with open(FAILED_POSTS_FILE, "a", encoding="utf-8") as f:
                f.write(f"{post_id} - {datetime.now()} - {e}\n")
            continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_json_path or os.path.join(OUTPUT_DIR, f"annotations_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.json_normalize(results)
    csv_path = output_csv_path or os.path.join(OUTPUT_DIR, f"annotations_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    print("Annotation Complete")
    print("JSON:", json_path)
    print("CSV :", csv_path)

# ======================================================
# ENTRY
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Post-only Reddit fraud annotation")
    parser.add_argument("--mode", choices=["interactive", "full", "test5", "test20"], default="interactive")
    parser.add_argument("--posts-file", default=POSTS_FILE)
    parser.add_argument("--comments-file", default=COMMENTS_FILE)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "full":
        run_annotation(
            posts_file=args.posts_file,
            comments_file=args.comments_file,
            output_csv_path=args.output_csv or None,
            output_json_path=args.output_json or None,
        )
    elif args.mode == "test5":
        run_annotation(
            limit=5,
            posts_file=args.posts_file,
            comments_file=args.comments_file,
            output_csv_path=args.output_csv or None,
            output_json_path=args.output_json or None,
        )
    elif args.mode == "test20":
        run_annotation(
            limit=20,
            posts_file=args.posts_file,
            comments_file=args.comments_file,
            output_csv_path=args.output_csv or None,
            output_json_path=args.output_json or None,
        )
    else:
        print("Post-Only Fraud Annotation System")
        print("1. Test on 5 posts")
        print("2. Annotate full dataset")
        print("3. Test on 20 posts")

        choice = input("Choose (1/2/3): ").strip()

        if choice == "1":
            run_annotation(limit=5)
        elif choice == "2":
            run_annotation()
        elif choice == "3":
            run_annotation(limit=20)
        elif choice == "4":
            post_id = input("Enter Post ID: ").strip()
            posts_df = pd.read_csv(POSTS_FILE)
            comments_df = pd.read_csv(COMMENTS_FILE)
            posts_df.columns = posts_df.columns.str.lower().str.strip()
            comments_df.columns = comments_df.columns.str.lower().str.strip()
            posts_df["post_id"] = posts_df["post_id"].astype(str)
            target = posts_df[posts_df["post_id"] == post_id]
            if target.empty:
                print("Post not found")
            else:
                print(f"Annotating specific post: {post_id}")
                print(annotate_post(target.iloc[0], comments_df))
        else:
            print("Invalid choice")
