import pandas as pd
import re

OUTPUT = "data/golden_set.csv"

df = pd.read_csv(OUTPUT)

def get_intent(text):
    text = str(text).lower()
    
    if any(kw in text for kw in ['play', 'stop', 'skip', 'buffer', 'pause', 'volume', 'sound']):
        return 'playback_issue'
    elif any(kw in text for kw in ['crash', 'freeze', 'bug', 'glitch', 'desktop', 'app', 'update', 'open', 'close', 'load']):
        return 'app_or_technical_issue'
    elif any(kw in text for kw in ['login', 'log in', 'password', 'account', 'access', 'hacked']):
        return 'account_or_access'
    elif any(kw in text for kw in ['premium', 'bill', 'charge', 'pay', 'subscription', 'refund', 'money']):
        return 'subscription_or_payment'
    elif any(kw in text for kw in ['missing', 'album', 'playlist', 'song', 'add', 'available', 'remove', 'download']):
        return 'music_or_content_request'
    elif any(kw in text for kw in ['how do', 'what is', 'feature', 'connect', 'how to']):
        return 'feature_or_product_question'
    else:
        return 'other_or_unclear'

def get_escalate(intent, text):
    text = str(text).lower()
    if intent in ['account_or_access', 'subscription_or_payment']:
        return 'yes'
    if any(kw in text for kw in ['angry', 'cancel', 'steal', 'hack', 'refund']):
        return 'yes'
    return 'no'

for idx, row in df.iterrows():
    intent = get_intent(row['message'])
    df.at[idx, 'intent'] = intent
    df.at[idx, 'escalate'] = get_escalate(intent, row['message'])

df.to_csv(OUTPUT, index=False)
print("Labeling complete!")
