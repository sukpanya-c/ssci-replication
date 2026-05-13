# Predictive Validity Notes

## Data sample used
- Selected market-facing genres: pop | rock-n-roll | hip-hop | jazz | electronic
- Analysis sample size: 2760

## Train/test split method
- GroupShuffleSplit with test_size=0.2 and random_state=42
- Grouping variable: track_id
- Leakage check: train/test overlap in track_id groups should be zero

## Model specifications
- genre_only: popularity ~ C(track_genre)
- genre_plus_pooled_audio: popularity ~ duration_ms + explicit + C(key) + C(mode) + C(time_signature) + C(track_genre) + danceability + energy + valence
- genre_conditioned_audio: popularity ~ duration_ms + explicit + C(key) + C(mode) + C(time_signature) + C(track_genre) * (danceability + energy + valence)

## Metric definitions
- RMSE: root mean squared error on the held-out test set
- MAE: mean absolute error on the held-out test set
- test R^2: out-of-sample coefficient of determination on the held-out test set
- train adjusted R^2: in-sample adjusted R^2 on the training set, reported as secondary context only

## Headline interpretation
Genre-conditioned feature models provide predictive, platform-facing evidence only. Relative to the pooled-audio model, the interaction-aware specification changes test R^2 by 0.029 and RMSE by 0.441. These comparisons support modest but useful predictive improvement for screening and content-positioning decisions, without implying causal effects or direct recommendation-system gains.
