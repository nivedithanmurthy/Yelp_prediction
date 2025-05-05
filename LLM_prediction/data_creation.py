import pandas as pd
from textblob import TextBlob

# ========================================
# 1. Load your files
# ========================================
business_df = pd.read_csv('business_prepared.csv')
reviews_df = pd.read_csv('review_prepared.csv')

# ========================================
# 2. Preprocessing
# ========================================
# Keep only necessary columns from reviews
reviews_df = reviews_df[['business_id', 'stars', 'text', 'date']]

# Make sure 'date' is parsed as datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# ========================================
# 3. Compute Sentiment Scores
# ========================================
def get_sentiment(text):
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity  # Returns value between -1 (bad) to 1 (good)
    except:
        return 0.0

reviews_df['sentiment'] = reviews_df['text'].apply(get_sentiment)

# ========================================
# 4. Build the test data
# ========================================
from datetime import datetime
import pandas as pd

# Generate list of the last 10 months
end_date = reviews_df['date'].max().replace(day=1)
month_list = pd.date_range(end=end_date, periods=10, freq='MS').strftime('%Y-%m').tolist()

test_rows = []

# For each business
for business_id in business_df['business_id'].unique():
    business_reviews = reviews_df[reviews_df['business_id'] == business_id].copy()

    # Extract year-month from date
    business_reviews['year_month'] = business_reviews['date'].dt.to_period('M').astype(str)

    # Group by month
    grouped = business_reviews.groupby('year_month').agg({
        'stars': 'mean',
        'text': lambda x: x.dropna().apply(lambda t: len(str(t).split())).mean(),
        'sentiment': 'mean'
    })

    # Reindex to include all 10 months, fill missing with 0
    grouped = grouped.reindex(month_list).fillna(0.0)

    test_rows.append({
        'business_id': business_id,
        'avg_ratings_last_10_months': grouped['stars'].tolist(),
        'avg_review_lengths_last_10_months': grouped['text'].tolist(),
        'avg_sentiments_last_10_months': grouped['sentiment'].tolist()
    })

# ========================================
# 5. Save to CSV
# ========================================
# Convert to DataFrame and save
test_df = pd.DataFrame(test_rows)
test_df.to_csv('test_data.csv', index=False)
print("test_data.csv created!")
