"""NLP Data Drift Detector evaluating text length distributions, vocabulary OOV, and statistical drift."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from scipy.stats import ks_2samp

from src.config import DATA_DIR, MODEL_DIR
from src.model.model_utils import load_artifact


def detect_drift(reference_mean: float, current_mean: float, threshold: float = 0.1) -> bool:
    """Basic scalar drift comparison (backward compatibility)."""
    return abs(reference_mean - current_mean) > threshold


class NLPDataDriftDetector:
    """Detects data drift in production NLP inference text against training reference baseline.
    
    Employs two-sample Kolmogorov-Smirnov statistical testing on input length distributions
    combined with Out-of-Vocabulary (OOV) rate analysis against the model's TF-IDF vocabulary.
    """

    def __init__(
        self,
        reference_profile_path: Optional[Path] = None,
        default_threshold: float = 0.20,
    ) -> None:
        self.reference_profile_path = reference_profile_path or (
            DATA_DIR / "reference" / "reference_profile.json"
        )
        self.default_threshold = default_threshold
        self._reference_profile: Optional[Dict[str, Any]] = None
        self._reference_vocab: Optional[Set[str]] = None

    def _get_reference_vocab(self) -> Set[str]:
        """Load feature vocabulary from the serialized TF-IDF vectorizer."""
        if self._reference_vocab is not None:
            return self._reference_vocab

        vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            try:
                vectorizer = load_artifact(vec_path)
                if hasattr(vectorizer, "vocabulary_"):
                    self._reference_vocab = set(vectorizer.vocabulary_.keys())
                    return self._reference_vocab
            except Exception:
                pass
        self._reference_vocab = set()
        return self._reference_vocab

    def _load_reference_profile(self) -> Dict[str, Any]:
        """Load pre-computed reference baseline statistics."""
        if self._reference_profile is not None:
            return self._reference_profile

        if self.reference_profile_path.exists():
            try:
                with open(self.reference_profile_path, "r", encoding="utf-8") as f:
                    self._reference_profile = json.load(f)
                    return self._reference_profile
            except Exception:
                pass

        # Fallback profile if json not available
        default_lengths = [100, 250, 400, 600, 850, 1100, 1400, 1800, 2200, 2800] * 50
        self._reference_profile = {
            "dataset_name": "IMDb Training Baseline (Default)",
            "sample_count": len(default_lengths),
            "mean_char_length": 1309.5,
            "mean_word_count": 230.2,
            "sample_char_lengths": default_lengths,
            "sample_word_counts": [l // 6 for l in default_lengths],
        }
        return self._reference_profile

    def calculate_drift(
        self,
        production_texts: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate recent production input texts for NLP data drift."""
        eval_threshold = threshold if threshold is not None else self.default_threshold
        ref = self._load_reference_profile()
        dataset_name = ref.get("dataset_name", "IMDb Reference Dataset")

        # Insufficient data check
        if not production_texts or len(production_texts) < 5:
            count = len(production_texts) if production_texts else 0
            return {
                "drift_detected": False,
                "metric": "Kolmogorov-Smirnov & Vocabulary OOV",
                "score": 0.0,
                "threshold": eval_threshold,
                "status": "INSUFFICIENT_DATA",
                "reference_dataset": dataset_name,
                "current_production_window": f"{count} recent production predictions (min 5 required)",
                "interpretation": "Insufficient production samples to perform statistical drift analysis.",
                "details": {
                    "production_sample_count": count,
                    "minimum_samples_required": 5,
                },
            }

        # 1. Text Length Analysis
        prod_char_lengths = [len(t) for t in production_texts]
        prod_word_counts = [len(t.split()) for t in production_texts]

        prod_mean_char = round(sum(prod_char_lengths) / len(prod_char_lengths), 2)
        prod_mean_words = round(sum(prod_word_counts) / len(prod_word_counts), 2)

        ref_lengths = ref.get("sample_char_lengths", [1000] * 50)
        ref_mean_char = ref.get("mean_char_length", 1300.0)

        # 2. Two-sample Kolmogorov-Smirnov test on character length distribution
        ks_stat, ks_pvalue = ks_2samp(prod_char_lengths, ref_lengths)
        ks_stat = round(float(ks_stat), 4)
        ks_pvalue = round(float(ks_pvalue), 4)

        # 3. Out-Of-Vocabulary (OOV) Rate Analysis
        vocab = self._get_reference_vocab()
        all_prod_tokens: List[str] = []
        for text in production_texts:
            tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
            all_prod_tokens.extend(tokens)

        if vocab and all_prod_tokens:
            oov_tokens = [t for t in all_prod_tokens if t not in vocab]
            oov_rate = round(len(oov_tokens) / len(all_prod_tokens), 4)
        else:
            oov_rate = 0.0

        # 4. Composite Drift Score
        # Drift score is the maximum of the KS length divergence and the vocabulary OOV rate
        drift_score = round(max(ks_stat, oov_rate), 4)
        drift_detected = drift_score >= eval_threshold

        status = "DRIFT_DETECTED" if drift_detected else "NORMAL"

        if drift_detected:
            reasons = []
            if ks_stat >= eval_threshold:
                reasons.append(f"text length divergence (KS={ks_stat} >= {eval_threshold})")
            if oov_rate >= eval_threshold:
                reasons.append(f"high out-of-vocabulary rate (OOV={oov_rate} >= {eval_threshold})")
            interpretation = f"Data drift detected due to: {', '.join(reasons)}."
        else:
            interpretation = (
                f"Production inputs match reference baseline within tolerance "
                f"(drift score {drift_score} < threshold {eval_threshold})."
            )

        return {
            "drift_detected": drift_detected,
            "metric": "Kolmogorov-Smirnov & Vocabulary OOV",
            "score": drift_score,
            "threshold": eval_threshold,
            "status": status,
            "reference_dataset": dataset_name,
            "current_production_window": f"{len(production_texts)} recent production predictions",
            "interpretation": interpretation,
            "details": {
                "production_sample_count": len(production_texts),
                "reference_sample_count": ref.get("sample_count", len(ref_lengths)),
                "production_mean_char_length": prod_mean_char,
                "reference_mean_char_length": ref_mean_char,
                "production_mean_word_count": prod_mean_words,
                "reference_mean_word_count": ref.get("mean_word_count", 230.0),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "oov_rate": oov_rate,
                "total_production_tokens_analyzed": len(all_prod_tokens),
            },
        }


# Global singleton drift detector
nlp_data_drift_detector = NLPDataDriftDetector()
