import pandas as pd
import requests

# Load your CSV
df = pd.read_csv("isarcasm_test.csv")

# Your Twitter Bearer Token (you need to generate this from developer portal)
BEARER_TOKEN = "1958471148494000128eren_yark"

def get_tweet_text(tweet_id, bearer_token=BEARER_TOKEN):
    """Fetch tweet text from Twitter API v2 given a tweet ID."""
    url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=text"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("data", {}).get("text", None)
    else:
        return None

# Replace tweet_id with actual text (if available)
texts = []
print("Fetching tweet texts... This may take a while depending on the number of tweets.")
for tid in df["tweet_id"]:
    tweet_text = get_tweet_text(tid)
    texts.append(tweet_text if tweet_text else f"UNAVAILABLE({tid})")

# Create a new column with text
df["tweet_text"] = texts

# Save new CSV
df.to_csv("tweets_with_text.csv", index=False)

print("Done! Output saved to tweets_with_text.csv")