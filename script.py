import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# 1️⃣ استخراج داده از وب‌سایت (نمونه: نقل‌قول‌های مثبت از Goodreads)
url = "https://quotes.toscrape.com"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# پیدا کردن نقل‌قول‌ها در صفحه
quotes = [quote.text.strip() for quote in soup.find_all("span", class_="text")]

# 2️⃣ تحلیل سنتیمنت با Vader
sia = SentimentIntensityAnalyzer()
sentiments = [sia.polarity_scores(quote)["compound"] for quote in quotes]

# 3️⃣ ذخیره خروجی در فایل CSV
df = pd.DataFrame({"Quote": quotes, "Sentiment Score": sentiments})
df.to_csv("sentiment_analysis_sample.csv", index=False)

print("✅ تحلیل سنتیمنت انجام شد! خروجی در فایل 'sentiment_analysis_sample.csv' ذخیره شد.")