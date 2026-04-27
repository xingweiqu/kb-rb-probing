"""Generate probing report."""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_report(
    probing_results_path: str,
    labels_path: str,
    output_path: str
) -> dict:
    """Generate comprehensive probing report."""
    # Load results
    with open(probing_results_path, encoding="utf-8") as f:
        results = json.load(f)

    # Load labels for distribution info
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    # Build report
    report = {
        "summary": {},
        "by_capability": {},
        "recommendations": []
    }

    for capability, layer_results in results.items():
        best_layer = max(layer_results, key=lambda k: layer_results[k])
        best_acc = layer_results[best_layer]

        # Get label distribution
        values = [l.get(capability) for l in labels.values() if capability in l]
        pos = values.count(True)
        neg = values.count(False)
        dropped = values.count(None)

        report["by_capability"][capability] = {
            "best_layer": best_layer,
            "best_accuracy": best_acc,
            "label_distribution": {
                "positive": pos,
                "negative": neg,
                "dropped": dropped,
                "total": len(values)
            },
            "significant": best_acc > 0.60  # Simple threshold
        }

        # Recommendations
        if best_acc < 0.55:
            report["recommendations"].append(
                f"{capability}: Low accuracy ({best_acc:.3f}). May need more data or better variants."
            )
        elif pos < 5 or neg < 5:
            report["recommendations"].append(
                f"{capability}: Imbalanced labels (pos={pos}, neg={neg}). Need more balanced data."
            )

    # Summary stats
    accuracies = [info["best_accuracy"] for info in report["by_capability"].values()]
    report["summary"] = {
        "total_capabilities": len(results),
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
        "max_accuracy": max(accuracies) if accuracies else 0,
        "min_accuracy": min(accuracies) if accuracies else 0,
        "significant_count": sum(1 for info in report["by_capability"].values() if info["significant"])
    }

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate HTML
    html_path = Path(output_path).with_suffix(".html")
    html = _generate_html(report)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("PROBING REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Total capabilities: {report['summary']['total_capabilities']}")
    logger.info(f"Mean accuracy: {report['summary']['mean_accuracy']:.3f}")
    logger.info(f"Significant capabilities: {report['summary']['significant_count']}")
    logger.info(f"\nBy capability:")
    for cap, info in sorted(report["by_capability"].items(), key=lambda x: -x[1]["best_accuracy"]):
        logger.info(f"  {cap:40s}: {info['best_accuracy']:.3f} (layer {info['best_layer']})")
    logger.info(f"{'='*60}\n")

    if report["recommendations"]:
        logger.info("Recommendations:")
        for rec in report["recommendations"]:
            logger.info(f"  - {rec}")

    logger.info(f"\nReport saved to {output_path}")
    logger.info(f"HTML report saved to {html_path}")

    return report


def _generate_html(report: dict) -> str:
    """Generate HTML report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Atomic Capability Probing Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .significant { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
    </style>
</head>
<body>
    <h1>Atomic Capability Probing Report</h1>

    <h2>Summary</h2>
    <ul>
"""
    html += f"        <li>Total capabilities: {report['summary']['total_capabilities']}</li>\n"
    html += f"        <li>Mean accuracy: {report['summary']['mean_accuracy']:.3f}</li>\n"
    html += f"        <li>Significant capabilities: {report['summary']['significant_count']}</li>\n"
    html += "    </ul>\n"

    html += """
    <h2>Results by Capability</h2>
    <table>
        <tr>
            <th>Capability</th>
            <th>Best Layer</th>
            <th>Accuracy</th>
            <th>Positive</th>
            <th>Negative</th>
            <th>Dropped</th>
        </tr>
"""

    for cap, info in sorted(report["by_capability"].items(), key=lambda x: -x[1]["best_accuracy"]):
        row_class = "significant" if info["significant"] else ""
        dist = info["label_distribution"]
        html += f"""        <tr class="{row_class}">
            <td>{cap}</td>
            <td>{info['best_layer']}</td>
            <td>{info['best_accuracy']:.3f}</td>
            <td>{dist['positive']}</td>
            <td>{dist['negative']}</td>
            <td>{dist['dropped']}</td>
        </tr>
"""

    html += """    </table>
"""

    if report["recommendations"]:
        html += """
    <h2>Recommendations</h2>
    <ul>
"""
        for rec in report["recommendations"]:
            html += f"        <li>{rec}</li>\n"
        html += "    </ul>\n"

    html += """
</body>
</html>
"""
    return html


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Probing results JSON")
    parser.add_argument("--labels", required=True, help="Atomic labels JSON")
    parser.add_argument("--output", required=True, help="Output report JSON")

    args = parser.parse_args()

    generate_report(
        probing_results_path=args.results,
        labels_path=args.labels,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
