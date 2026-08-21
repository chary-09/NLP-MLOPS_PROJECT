"""Write human-readable Phase 1 evaluation reports from saved metrics."""

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT_DIR / "data" / "models" / "metrics.json"
REPORT_PATH = ROOT_DIR / "reports" / "model_comparison_report.md"


def main() -> None:
	metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
	REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
	lines = [
		"# IMDb Sentiment Model Comparison",
		"",
		f"Best model: **{metrics['best_model']}**",
		"",
		"| Model | Accuracy | Precision | Recall | F1 Score |",
		"|---|---:|---:|---:|---:|",
	]
	for name, values in metrics["models"].items():
		lines.append(
			f"| {name} | {values['accuracy']:.4f} | {values['precision']:.4f} | "
			f"{values['recall']:.4f} | {values['f1_score']:.4f} |"
		)
	lines.extend(["", "## Confusion Matrices", ""])
	for name, values in metrics["models"].items():
		lines.extend([f"### {name}", "", "```text", str(values["confusion_matrix"]), "```", ""])
	REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
	print(f"Evaluation report saved to {REPORT_PATH}")


if __name__ == "__main__":
	main()
