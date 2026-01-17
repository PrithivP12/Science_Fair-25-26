#!/usr/bin/env python3
"""
Generate QC report from feature data.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import apply_qc_filters
from src.schema import QCRules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_qc_report(df: pd.DataFrame, output_path: Path):
    """Generate HTML QC report."""
    passed_df, rejected_df = apply_qc_filters(df, log_rejections=False)
    
    # Calculate statistics
    total = len(df)
    passed = len(passed_df)
    rejected = len(rejected_df)
    
    # Rejection reasons
    rejection_reasons = {}
    for _, row in rejected_df.iterrows():
        reasons = row['reasons'].split('; ')
        for reason in reasons:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QC Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .stat {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
            .passed {{ color: green; font-weight: bold; }}
            .rejected {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Quality Control Report</h1>
        
        <div class="stat">
            <h2>Summary</h2>
            <p>Total samples: {total}</p>
            <p class="passed">Passed: {passed} ({passed/total*100:.1f}%)</p>
            <p class="rejected">Rejected: {rejected} ({rejected/total*100:.1f}%)</p>
        </div>
        
        <div class="stat">
            <h2>Rejection Reasons</h2>
            <table>
                <tr><th>Reason</th><th>Count</th><th>Percentage</th></tr>
    """
    
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        html += f"<tr><td>{reason}</td><td>{count}</td><td>{count/total*100:.1f}%</td></tr>\n"
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info(f"QC report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate QC report")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output HTML report")
    args = parser.parse_args()
    
    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    generate_qc_report(df, Path(args.output))


if __name__ == "__main__":
    main()

