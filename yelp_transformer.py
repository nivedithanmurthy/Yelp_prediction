import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os

# Load the data
reviews = pd.read_csv("review_prepared.csv")
businesses = pd.read_csv("business_prepared.csv")

print(f"Loaded {len(reviews)} reviews")
print(f"Loaded {len(businesses)} businesses")

# Process the data
print(f"Using all {len(reviews)} reviews")
reviews = reviews.groupby('business_id').apply(
    lambda x: x.sample(n=min(100, len(x)), random_state=42)
).reset_index(drop=True)

# Monthly aggregation
reviews['date'] = pd.to_datetime(reviews['date'])
reviews['month'] = reviews['date'].dt.to_period('M')

monthly_grouped = (
    reviews.groupby(['business_id', 'month'])
    .agg({
        'stars': 'mean',
        'review_id': 'count',
        'text': list
    })
    .reset_index()
    .rename(columns={
        'stars': 'monthly_avg_rating',
        'review_id': 'monthly_review_count'
    })
)

# Sentiment Analysis function
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(x):
    if isinstance(x, str):
        text = x
    elif x is None or (isinstance(x, (float, int)) and np.isnan(x)):
        text = ""
    else:
        text = str(x)
    return analyzer.polarity_scores(text)

monthly_grouped['sentiment'] = monthly_grouped['text'].apply(get_sentiment)

# Merge with business info
merged_df = monthly_grouped.merge(
    businesses[['business_id', 'business_latitude', 'business_longitude', 'business_categories', 'business_open']],
    on='business_id'
)

# Define Dataset
class YelpDataset(Dataset):
    def __init__(self, data, business_ids):
        self.data = data
        self.business_ids = business_ids
        self.seq_len = 10
        self.data_subset = data[data['business_id'].isin(self.business_ids)]

    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, idx):
        business_id = self.data_subset.iloc[idx]['business_id']
        group = self.data_subset[self.data_subset['business_id'] == business_id]
        group = group.sort_values("month")

        if len(group) < self.seq_len + 1:
            padding = pd.DataFrame([group.iloc[-1].to_dict()] * (self.seq_len + 1 - len(group)))
            group = pd.concat([group, padding], ignore_index=True)

        sentiment_values = group['sentiment'].apply(lambda x: x.get('compound', 0) if isinstance(x, dict) else 0)
        sentiment_values = sentiment_values[:self.seq_len]
        sentiment_values = sentiment_values.fillna(0)

        x_seq = group.iloc[:self.seq_len][['monthly_avg_rating', 'monthly_review_count']].copy()
        x_seq['sentiment'] = sentiment_values.values

        x_seq = x_seq.apply(pd.to_numeric, errors='coerce')
        x_seq = x_seq.fillna(0)
        x_seq = x_seq.values

        y = {
            'closure': torch.tensor(group['business_open'].iloc[-1], dtype=torch.float32),
            'rating': torch.tensor(group['monthly_avg_rating'].iloc[-1], dtype=torch.float32),
            'review_count': torch.tensor(group['monthly_review_count'].iloc[-1], dtype=torch.float32),
            'popularity': torch.tensor(group['monthly_review_count'].iloc[-1] * sentiment_values.iloc[-1], dtype=torch.float32)
        }

        return torch.tensor(x_seq, dtype=torch.float32), y, business_id

# Define Model
class MultiTaskTransformer(pl.LightningModule):
    def __init__(self, input_dim=3, embed_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.cls_head = nn.Linear(embed_dim, 1)
        self.rating_head = nn.Linear(embed_dim, 1)
        self.count_head = nn.Linear(embed_dim, 1)
        self.popularity_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x[-1]
        return {
            'closure': torch.sigmoid(self.cls_head(x)),
            'rating': self.rating_head(x),
            'review_count': self.count_head(x),
            'popularity': self.popularity_head(x)
        }

class YelpForecaster(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MultiTaskTransformer()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss_cls = nn.BCELoss()(out['closure'].squeeze(), y['closure'])
        loss_rating = nn.MSELoss()(out['rating'].squeeze(), y['rating'])
        loss_count = nn.MSELoss()(out['review_count'].squeeze(), y['review_count'])
        loss_pop = nn.MSELoss()(out['popularity'].squeeze(), y['popularity'])
        loss = loss_cls + loss_rating + loss_count + loss_pop
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# Train/Test Split
business_ids = merged_df['business_id'].unique()
train_ids, test_ids = train_test_split(business_ids, test_size=0.2, random_state=42)

train_dataset = YelpDataset(merged_df, train_ids)
test_dataset = YelpDataset(merged_df, test_ids)

# Corrected custom collate
def custom_collate(batch):
    xs, ys, business_ids = zip(*batch)
    ys_stacked = {
        'closure': torch.stack([y['closure'] for y in ys]),
        'rating': torch.stack([y['rating'] for y in ys]),
        'review_count': torch.stack([y['review_count'] for y in ys]),
        'popularity': torch.stack([y['popularity'] for y in ys]),
    }
    return torch.stack(xs), ys_stacked

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

model = YelpForecaster()

trainer = pl.Trainer(max_epochs=10, accelerator='auto')
trainer.fit(model, train_loader)

# Evaluation
model.eval()

closures_true = []
closures_pred = []
ratings_true = []
ratings_pred = []
review_counts_true = []
review_counts_pred = []
popularity_true = []
popularity_pred = []

for x, y in test_loader:
    with torch.no_grad():
        preds = model(x)
    closures_true.extend(y['closure'].numpy())
    closures_pred.extend((preds['closure'].squeeze() > 0.5).float().numpy())
    ratings_true.extend(y['rating'].numpy())
    ratings_pred.extend(preds['rating'].squeeze().numpy())
    review_counts_true.extend(y['review_count'].numpy())
    review_counts_pred.extend(preds['review_count'].squeeze().numpy())
    popularity_true.extend(y['popularity'].numpy())
    popularity_pred.extend(preds['popularity'].squeeze().numpy())

# Classification Metrics
accuracy = accuracy_score(closures_true, closures_pred)
precision = precision_score(closures_true, closures_pred)
recall = recall_score(closures_true, closures_pred)
f1 = f1_score(closures_true, closures_pred)

# Regression Metrics
mse_rating = mean_squared_error(ratings_true, ratings_pred)
mae_rating = mean_absolute_error(ratings_true, ratings_pred)

mse_count = mean_squared_error(review_counts_true, review_counts_pred)
mae_count = mean_absolute_error(review_counts_true, review_counts_pred)

mse_popularity = mean_squared_error(popularity_true, popularity_pred)
mae_popularity = mean_absolute_error(popularity_true, popularity_pred)

print("\n=== Test Set Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Rating MSE: {mse_rating:.4f}, MAE: {mae_rating:.4f}")
print(f"Review Count MSE: {mse_count:.4f}, MAE: {mae_count:.4f}")
print(f"Popularity MSE: {mse_popularity:.4f}, MAE: {mae_popularity:.4f}")

# Future Predictions
def categorize_closure(prob):
    if prob < 0.3:
        return "Grow"
    elif prob < 0.7:
        return "Stable"
    else:
        return "Decline"

future_preds = []

for i in range(len(test_dataset)):
    sample_input, _, business_id = test_dataset[i]
    sample_input = sample_input.unsqueeze(0)

    with torch.no_grad():
        preds = model(sample_input)

    closure_prob = preds['closure'].item()
    closure_category = categorize_closure(closure_prob)

    future_preds.append({
        'business_id': business_id,
        'closure_prob': closure_prob,
        'closure_category': closure_category,
        'predicted_rating': preds['rating'].item(),
        'predicted_review_count': preds['review_count'].item(),
        'predicted_popularity': preds['popularity'].item()
    })

future_df = pd.DataFrame(future_preds)
future_df.to_csv("future_predictions_with_category.csv", index=False)
print("\nFuture predictions (with closure categories) saved to 'future_predictions_with_category.csv' ")
