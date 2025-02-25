import argparse
from evaluation import parse_tsv_file, calculate_metrics

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate NER model predictions against a gold standard.")
    parser.add_argument("--gold_standard", required=True, help="Path to the gold standard TSV file.")
    parser.add_argument("--predictions", required=True, help="Path to the predictions TSV file.")
    args = parser.parse_args()

    # Load TSV files
    gs = parse_tsv_file(args.gold_standard)
    pred = parse_tsv_file(args.predictions)

    # Calculate metrics
    P_samples, P, R_samples, R, F1_samples, F1 = calculate_metrics(gs, pred)

    # Print results
    print("Micro-average Precision:", P)
    print("Micro-average Recall:", R)
    print("Micro-average F1 score:", F1)

if __name__ == "__main__":
    main()