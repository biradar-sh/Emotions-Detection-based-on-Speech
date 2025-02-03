# Emotions Detection Based on Speech

## Authors:

Akshay Krishnan | Shruti Biradar | Siva Akhil Kumar Reddy Nukala

## Objective:

Understanding human emotions from speech can provide valuable insights into user opinions about topics, brands, products, events, or individuals. This project aims to detect emotions in text data using multi-class classification algorithms. Key applications include sentiment analysis, customer service enhancement, brand reputation monitoring, cyberbullying detection, and threat identification. The model performance is evaluated using the Macro F1 score, which is beneficial for handling class imbalances.

## Datasets:

We used three open datasets:

1. **SMILE Twitter Emotion Dataset** - 3,079 observations with 14 emotions (filtered to ensure balanced classes).
2. **Emotions in Text (Kaggle Dataset)** - 21,459 observations covering 6 emotions.
3. **Hugging Face Dair-AI Emotion Dataset** - 20,000 observations covering 6 emotions.

After merging these datasets, our final dataset contains 44,538 observations with the following distribution:

- Happy: 14,927
- Sad: 12,108
- Angry: 5,766
- Fear: 5,025
- Love: 3,282
- Neutral: 2,505
- Surprise: 925

### Project Goals:

1. Identify key words differentiating emotions.
2. Clean dataset by removing usernames and links.
3. Test multiple machine learning models and select the best.

## Data Preprocessing & Analysis:

### Data Preprocessing:

- Removed usernames, links, punctuation, diacritics, extra spaces, and numbers.
- Expanded contractions and corrected spelling.
- Applied lemmatization to reduce word variations.
- Used TF-IDF vectorization for ML models.
- Used tokenization and sequence padding for neural networks.
- Integrated GloVe embeddings for Bidirectional LSTM models.

### Exploratory Data Analysis:

- **Emotion Distribution**: Addressed imbalance with stratified k-fold cross-validation.
- **Tweet Length Distribution**: Most tweets were under 50 characters.
- **Word Clouds**: Identified key words associated with each emotion.

## Modeling:

### Models Used:

1. **XGBoost** - Gradient boosting algorithm optimizing loss function.
2. **CATBoost** - Boosting algorithm with ordered boosting to prevent target leakage.
3. **LightGBM** - Tree-based learning growing leaf-wise for efficiency.
4. **RoBERTa** - Pretrained deep learning model optimized for NLP tasks.

### Hyperparameters:

| Model    | Hyperparameters                                                                                     |
| -------- | --------------------------------------------------------------------------------------------------- |
| XGBoost  | max\_depth: 10, learning\_rate: 0.05, n\_estimators: 500, reg\_lambda: 0.5, gamma: 1.0, alpha: 0.01 |
| CATBoost | learning\_rate: 0.16, colsample\_bylevel: 0.7, iterations: 800, max\_depth: 5, l2\_leaf\_reg: 8     |
| LightGBM | learning\_rate: 0.2, max\_depth: 10, n\_estimators: 500                                             |
| RoBERTa  | vocab\_size: 30522, hidden\_size: 768, num\_hidden\_layers: 12, num\_attention\_heads: 12           |

### Model Results:

| Model    | Train Accuracy | Test Accuracy | Train Macro F1 | Test Macro F1 |
| -------- | -------------- | ------------- | -------------- | ------------- |
| XGBoost  | 0.68           | 0.55          | 0.66           | 0.49          |
| CATBoost | 0.56           | 0.52          | 0.44           | 0.40          |
| LightGBM | 0.84           | 0.66          | 0.82           | 0.56          |
| RoBERTa  | **0.94**       | **0.92**      | **0.87**       | **0.83**      |

RoBERTa outperformed all other models and was selected as the final model.

## Model Inference:

Example Input:

```plaintext
"Media knowingly doesn’t tell the truth. A great danger to our country. The failing New York Times has become a joke."




