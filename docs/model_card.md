# Model Card

## Intended Use

The project compares recurrent neural network architectures for movie review sentiment modelling. The preferred final use case is binary sentiment classification for quick review interpretation.

## Architectures

The original notebook trains:

- SimpleRNN classification models.
- LSTM classification models.
- GRU classification models.
- SimpleRNN regression models.
- LSTM regression models.
- GRU regression models.

Each architecture is compared with augmented and non-augmented training data.

## Inputs

Tokenised and padded movie review text.

## Outputs

Classification models output a binary sentiment probability.

Regression models output a continuous score prediction.

## Metrics

Classification:

- Accuracy.
- Confusion matrix.
- Precision, recall, and F1 report.

Regression:

- MAE.
- MSE.
- RMSE.
- MAPE.
- R2.

## Limitations

- RNN models may underperform transformer-based models on complex multilingual sentiment.
- Augmentation quality depends on external paraphrase and translation models.
- The notebook was designed for an academic Colab workflow and may need path changes for local training.
- The data should not be assumed representative of all movie review styles or languages.

## Ethical Considerations

Sentiment predictions are model estimates and should not be used as the sole basis for decisions affecting people. Review language, sarcasm, mixed sentiment, and translation quality can change predictions.
