import requests
import time
import csv
import json
import os
import random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue

# ================== CONSTANTS ==================

BASE_URL = "https://www.reddit.com"
HEADERS = {
    "User-Agent": "academic-research/1.0 (reddit scam detection study)"
}

NOW_UTC = int(datetime.now(tz=timezone.utc).timestamp())
THREE_MONTHS_SEC = 90 * 24 * 3600 * 4 * 2    # last 3 months

CHECKPOINT_EVERY = 10

# Thread pool sizes
MAX_SUBREDDIT_WORKERS = 3  # Process 3 subreddits concurrently
MAX_COMMENT_WORKERS = 5     # Fetch comments for 5 posts concurrently

# ================== THREAD-SAFE LOGGING ==================

log_lock = Lock()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    with log_lock:
        print(f"[{ts}] {msg}", flush=True)

# ================== UTILS ==================

def safe_sleep(base, reason=""):
    sleep_time = random.uniform(base, base + 1.5)
    if reason:
        log(f"Sleeping {sleep_time:.2f}s ({reason})")
    time.sleep(sleep_time)

def guarded_request(url, params=None, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)

            if r.status_code == 200:
                return r

            if r.status_code in (403, 429):
                log(f"HTTP {r.status_code} (attempt {attempt}/{retries})")
                backoff = 60 * attempt
                log(f"Backing off {backoff}s")
                time.sleep(backoff)
            else:
                r.raise_for_status()
        except Exception as e:
            log(f"Request error: {e}")
            if attempt < retries:
                time.sleep(10 * attempt)

    log("Request failed after retries, skipping.")
    return None

# ================== CACHE FOR DUPLICATE AVOIDANCE ==================

CACHE_FILE = "scrape_cache.json"
cache_lock = Lock()
CACHE_SET = set()

def load_cache():
    global CACHE_SET
    if not os.path.exists(CACHE_FILE):
        CACHE_SET = set()
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                CACHE_SET = set(data)
            else:
                CACHE_SET = set()
    except Exception as e:
        log(f"Could not load cache {CACHE_FILE}: {e}")
        CACHE_SET = set()

def save_cache():
    tmp = f"{CACHE_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(CACHE_SET), f, indent=2)
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        log(f"Failed saving cache: {e}")


# ================== KEYWORD FILTERING ==================

def build_post_keyword_set(keywords):
    groups = [
        "general_scam_indicators",
        "crypto_scams",
        "investment_scams",
        "upi_payment_fraud",
        "credit_card_banking_fraud",
        "impersonation_scams"
    ]

    kw_set = set()
    for g in groups:
        for kw in keywords.get(g, []):
            kw_set.add(kw.lower())

    return kw_set

def post_matches_keywords(title, body, keyword_set):
    text = f"{title} {body}".lower()
    return any(kw in text for kw in keyword_set)

# ================== WEAK LABELS ==================

def weak_label_score(text, keywords):
    text = text.lower()
    score = 0

    for p in keywords["comment_confirmation_strong"]:
        if p in text:
            score += 3
    for p in keywords["comment_confirmation_medium"]:
        if p in text:
            score += 1
    for p in keywords["comment_negative_signals"]:
        if p in text:
            score -= 2

    return score

# ================== POST COLLECTION ==================

def fetch_posts(subreddit, max_posts, sleep_time, post_keyword_set):
    """Generator that yields posts matching criteria"""
    yielded_count = 0
    after = None
    page = 0

    log(f"Starting r/{subreddit}")

    while yielded_count < max_posts:
        page += 1
        log(f"r/{subreddit} - Fetching page {page} (yielded: {yielded_count})")

        url = f"{BASE_URL}/r/{subreddit}/new.json"
        params = {"limit": 100, "after": after}

        r = guarded_request(url, params)
        if not r:
            break

        try:
            data = r.json()["data"]
            children = data["children"]
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            log(f"Error parsing response for r/{subreddit}: {e}")
            break

        if not children:
            log(f"r/{subreddit} - No more posts returned")
            break

        for item in children:
            d = item["data"]
            created = d["created_utc"]

            # TIME WINDOW: LAST 3 MONTHS
            if created < NOW_UTC - THREE_MONTHS_SEC:
                log(f"r/{subreddit} - Reached posts older than 3 months, stopping")
                return

            title = d["title"]
            body = d["selftext"]

            # KEYWORD FILTER
            if not post_matches_keywords(title, body, post_keyword_set):
                continue

            post_data = {
                "post_id": d["id"],
                "subreddit": subreddit,
                "title": title,
                "body_text": body,
                "created_utc": created,
                "score": d["score"],
                "num_comments": d["num_comments"],
                "link_flair": d.get("link_flair_text")
            }

            yield post_data
            yielded_count += 1

            age_days = (NOW_UTC - created) // 86400
            log(f"r/{subreddit} - Accepted post {d['id']} (age {age_days} days)")

            if yielded_count >= max_posts:
                log(f"r/{subreddit} - Reached max post cap")
                return

        after = data.get("after")
        if not after:
            log(f"r/{subreddit} - Pagination cursor exhausted")
            break

        safe_sleep(sleep_time, f"r/{subreddit} post pagination")

# ================== COMMENT COLLECTION ==================

def fetch_comments(post_id, subreddit, top_n, keywords, sleep_time):
    """Fetch comments for a single post"""
    log(f"Fetching top {top_n} comments for post {post_id}")

    url = f"{BASE_URL}/comments/{post_id}.json"
    params = {"limit": top_n, "sort": "top"}

    r = guarded_request(url, params)
    if not r:
        return []

    comments = []
    
    try:
        children = r.json()[1]["data"]["children"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        log(f"Could not parse comments for {post_id}, skipping.")
        return []

    for item in children:
        if item["kind"] != "t1":
            continue

        d = item["data"]
        wl = weak_label_score(d.get("body", ""), keywords)

        comments.append({
            "comment_id": d["id"],
            "post_id": post_id,
            "subreddit": subreddit,
            "comment_text": d.get("body", ""),
            "comment_score": d["score"],
            "comment_created_utc": d["created_utc"],
            "weak_label_score": wl
        })

        if len(comments) >= top_n:
            break

    safe_sleep(sleep_time, "comment fetch")
    return comments

# ================== THREAD-SAFE FILE WRITERS ==================

class ThreadSafeCSVWriter:
    """Thread-safe CSV writer with file locking"""
    
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.lock = Lock()
        self.file = None
        self.writer = None
        
    def open(self):
        mode = "w"
        write_header = True
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            mode = "a"
            write_header = False

        self.file = open(self.filepath, mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()
        
    def writerow(self, row):
        with self.lock:
            self.writer.writerow(row)
            
    def flush(self):
        with self.lock:
            self.file.flush()
            
    def close(self):
        if self.file:
            self.file.close()

# ================== CONCURRENT PROCESSING ==================

def process_post_with_comments(post, params, keywords, post_writer, comment_writer, stats):
    """Process a single post and its comments"""
    try:
        # Skip if already seen in cache
        post_id = post.get("post_id")
        with cache_lock:
            if post_id in CACHE_SET:
                log(f"Skipping cached post {post_id}")
                return False
        
        # Write post
        post_writer.writerow(post)
        
        # Fetch comments concurrently would be overkill, just fetch them
        comments = fetch_comments(
            post["post_id"],
            post["subreddit"],
            params["top_n_comments"],
            keywords,
            params["sleep_time_sec"]
        )
        
        # Write all comments
        for c in comments:
            comment_writer.writerow(c)
        
        # Update stats
        with stats['lock']:
            stats['posts_processed'] += 1
            stats['comments_collected'] += len(comments)
            
            if stats['posts_processed'] % CHECKPOINT_EVERY == 0:
                post_writer.flush()
                comment_writer.flush()
                log(f"CHECKPOINT @ {stats['posts_processed']} posts, {stats['comments_collected']} comments")
        
        # Mark as seen and persist cache
        with cache_lock:
            CACHE_SET.add(post_id)
            try:
                save_cache()
            except Exception:
                log(f"Warning: failed to save cache for post {post_id}")

        return True
    except Exception as e:
        log(f"Error processing post {post.get('post_id', 'unknown')}: {e}")
        return False

def process_subreddit(subreddit, params, keywords, post_keyword_set, post_writer, comment_writer, stats):
    """Process all posts from a single subreddit with concurrent comment fetching"""
    log(f"===== ENTER r/{subreddit} =====")
    
    post_generator = fetch_posts(
        subreddit,
        params["max_results"],
        params["sleep_time_sec"],
        post_keyword_set
    )
    
    # Collect posts into batches for concurrent comment processing
    post_batch = []
    batch_size = MAX_COMMENT_WORKERS
    
    for post in post_generator:
        post_batch.append(post)
        
        # Process batch when full
        if len(post_batch) >= batch_size:
            with ThreadPoolExecutor(max_workers=MAX_COMMENT_WORKERS) as executor:
                futures = []
                for p in post_batch:
                    future = executor.submit(
                        process_post_with_comments,
                        p, params, keywords, post_writer, comment_writer, stats
                    )
                    futures.append(future)
                
                # Wait for all to complete
                for future in as_completed(futures):
                    future.result()
            
            post_batch = []
    
    # Process remaining posts
    if post_batch:
        with ThreadPoolExecutor(max_workers=MAX_COMMENT_WORKERS) as executor:
            futures = []
            for p in post_batch:
                future = executor.submit(
                    process_post_with_comments,
                    p, params, keywords, post_writer, comment_writer, stats
                )
                futures.append(future)
            
            for future in as_completed(futures):
                future.result()
    
    log(f"===== EXIT r/{subreddit} =====")

# ================== PIPELINE ==================

def run(config_path="config.json"):
    log("Loading configuration")

    with open(config_path) as f:
        config = json.load(f)

    subreddits = (
        config["subreddits"]["tier_1"] +
        config["subreddits"]["tier_2"] +
        config["subreddits"]["tier_3"]
    )

    params = config["collection_params"]
    keywords = config["keywords"]

    post_keyword_set = build_post_keyword_set(keywords)

    log(f"Loaded {len(post_keyword_set)} scam keywords")
    log(f"Mode: LAST 3 MONTHS + KEYWORD FILTERING")
    log(f"Concurrency: {MAX_SUBREDDIT_WORKERS} subreddits, {MAX_COMMENT_WORKERS} comment threads per subreddit")

    # Thread-safe writers
    post_writer = ThreadSafeCSVWriter("posts.csv", [
        "post_id", "subreddit", "title", "body_text",
        "created_utc", "score", "num_comments", "link_flair"
    ])
    comment_writer = ThreadSafeCSVWriter("comments.csv", [
        "comment_id", "post_id", "subreddit",
        "comment_text", "comment_score",
        "comment_created_utc", "weak_label_score"
    ])

    post_writer.open()
    comment_writer.open()

    # Load cache of already-seen post ids
    load_cache()

    # Shared statistics
    stats = {
        'posts_processed': 0,
        'comments_collected': 0,
        'lock': Lock()
    }

    try:
        # Process subreddits concurrently
        with ThreadPoolExecutor(max_workers=MAX_SUBREDDIT_WORKERS) as executor:
            futures = []
            for sub in subreddits:
                future = executor.submit(
                    process_subreddit,
                    sub, params, keywords, post_keyword_set,
                    post_writer, comment_writer, stats
                )
                futures.append(future)
            
            # Wait for all subreddits to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log(f"Subreddit processing error: {e}")

        # Final flush
        post_writer.flush()
        comment_writer.flush()
        
        log("=" * 60)
        log(f"SCRAPING COMPLETE")
        log(f"Total posts: {stats['posts_processed']}")
        log(f"Total comments: {stats['comments_collected']}")
        log("=" * 60)

    finally:
        post_writer.close()
        comment_writer.close()

# ================== ENTRY ==================

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log("Interrupted by user — data saved up to last checkpoint")
    except Exception as e:
        log(f"Fatal error: {e}")
        raise