import pandas as pd
import time
import json
from openai import OpenAI
import os

# ===========================
# 1. SETUP
# ===========================

# Set your OpenAI API key
client = OpenAI(api_key="your_API_key")

# Load your test data
test_data = pd.read_csv("test_data_LLM.csv")

# ===========================
# 2. HELPER FUNCTION TO CREATE PROMPT
# ===========================

def create_business_prompt(ratings, reviews, sentiments):
    prompt = f"""
Business Summary:
- Last 10 months average star ratings: {ratings}
- Last 10 months review counts: {reviews}
- Last 10 months sentiment scores: {sentiments}

Questions:
1. Business growth category (grow or stable or decline)
2. Predict next month's average star rating (only give number).
3. Predict next month's review count (only give number).
4. Predict next month's popularity (review count × sentiment score).
5. Provide 2-3 suggestions to help the business improve its rating, customer satisfaction, or popularity.

IMPORTANT:
Respond in JSON format:
{{
  "closure": "...",
  "predicted_rating": ...,
  "predicted_review_count": ...,
  "predicted_popularity": ...,
  "suggestions": "..."
}}
    """.strip()
    return prompt

# ===========================
# 3. HELPER FUNCTION TO CALL OPENAI
# ===========================

def query_openai(prompt, model="gpt-4", temperature=0.3):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert business forecast analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return None

# ===========================
# 4. MAIN LOOP (LIMITED TO 10 BUSINESSES)
# ===========================

predictions = []

# Limit to first 10 rows
test_data_subset = test_data.head(10)

for idx, row in test_data_subset.iterrows():
    business_id = row['business_id']
    ratings = eval(row['ratings'])        # Convert string to list
    reviews = eval(row['reviews'])
    sentiments = eval(row['sentiments'])
    
    # Create the prompt
    prompt = create_business_prompt(ratings, reviews, sentiments)
    
    # Query LLM
    print(f"Predicting for business_id: {business_id}...")
    output = query_openai(prompt)
    
    if output is None:
        continue

    # Parse JSON output
    try:
        prediction = json.loads(output)
        prediction['business_id'] = business_id
        predictions.append(prediction)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for business_id: {business_id}")
        continue

    # Sleep to avoid rate limits (adjust if needed)
    time.sleep(20)


# ===========================
# 5. SAVE RESULTS
# ===========================

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions)

# Save to CSV
predictions_df.to_csv("llm_predictions3.csv", index=False)

print("Done! Predictions saved to 'llm_predictions3.csv'.")
