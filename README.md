# Named Entity Recognition (NER) Model Evaluation

This repository contains a Python script (`main.py`) for evaluating the performance of a Named Entity Recognition (NER) model. The evaluation is based on comparing the model's predictions against a gold standard (reference) in BRAT format, stored as TSV files.

## Requirements

- Python 3.7 or higher
- pandas (`pip install pandas`)

## Input Format

The evaluation script requires two TSV files as input:
1. **Gold Standard (Reference) File**: Contains the ground truth annotations in BRAT format.
2. **Predicted File**: Contains the model's predictions in BRAT format.

Both files must follow the same structure, as shown below:

### Example of a TSV File in BRAT Format

| filename                      | ann_id | label          | start_span | end_span | text                          |
|-------------------------------|--------|----------------|------------|----------|-------------------------------|
| es-S0210-56912007000900007-3  | T1     | PROCEDIMIENTO  | 609        | 643      | tomografía computarizada abdominal |
| es-S0210-56912007000900007-3  | T2     | PROCEDIMIENTO  | 799        | 824      | intubada orotraquealmente      |
| es-S0210-56912007000900007-3  | T3     | PROCEDIMIENTO  | 827        | 861      | conectada a un respirador mecánico |

### Columns Description
- **filename**: The name of the document or file where the annotation appears.
- **ann_id**: The annotation ID (e.g., `T1`, `T2`).
- **label**: The entity label (e.g., `PROCEDIMIENTO`).
- **start_span**: The starting character offset of the entity in the text.
- **end_span**: The ending character offset of the entity in the text.
- **text**: The actual text of the entity.

## Usage

1. Place your gold standard TSV file and predicted TSV file in the same directory as `main.py`.
2. Run the script using the following command:
   ```bash
   python main.py --gold_standard <gold_standard.tsv> --predictions <predictions.tsv> [--entities ENTITY1 ENTITY2 ...]
   ```

   Example with Entities filter:
   ```bash 
   python main.py --gold_standard ./example_data/ner-groundtruth.tsv --predictions ./example_data/ner-prediction.tsv --entities enfermedad procedure 
   ```

   Example without entities filter:
   ```bash 
   python main.py --gold_standard ./example_data/ner-groundtruth.tsv --predictions ./example_data/ner-prediction.tsv 
   ```


Replace `<gold_standard.tsv>` and `<predictions.tsv>` with the paths to your input files.

Use the optional `--entities` argument to specify a list of entities to evaluate. If not provided, all entities will be evaluated.

3. The script will output the following metrics:
   - **Precision (P)**: Micro-average precision.
   - **Recall (R)**: Micro-average recall.
   - **F1 Score (F1)**: Micro-average F1 score.
   - **Precision, Recall, and F1 per clinical case**: Metrics calculated for each document (clinical case).

## Notes
- Ensure that the `filename`, `start_span`, `end_span`, and `label` columns in both files match exactly.
- The script assumes that there are no overlapping annotations in the input files.
- If there are duplicated entries in the input files, the script will automatically remove them and issue a warning.
