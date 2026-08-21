# IMDb Sentiment Model Comparison

Best model: **logistic_regression**

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| logistic_regression | 0.8904 | 0.8906 | 0.8904 | 0.8904 |
| naive_bayes | 0.8597 | 0.8599 | 0.8597 | 0.8597 |
| linear_svm | 0.8904 | 0.8906 | 0.8904 | 0.8904 |

## Confusion Matrices

### logistic_regression

```text
[[3298, 452], [370, 3380]]
```

### naive_bayes

```text
[[3182, 568], [484, 3266]]
```

### linear_svm

```text
[[3298, 452], [370, 3380]]
```
