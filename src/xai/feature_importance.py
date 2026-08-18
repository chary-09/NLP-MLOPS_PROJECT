def top_features(model, vectorizer, limit: int = 10) -> list[dict]:
    names = vectorizer.get_feature_names_out()
    weights = model.coef_.max(axis=0)
    order = weights.argsort()[-limit:][::-1]
    return [{"feature": str(names[i]), "importance": float(weights[i])} for i in order]
