# AWS MLOps (SageMaker) – Decision Tree Starter

End-to-end sample with SageMaker Pipelines using a **DecisionTreeClassifier**, supporting two datasets:
- Iris (built-in) and
- **Cars** (numeric-only CSV included).

## Quickstart (local)

```bash
pip install -r requirements.txt

# Use cars dataset
python src/preprocess.py --dataset cars --cars-csv ./data/cars.csv --output-base ./processing
python src/train.py --train-path ./processing/train/train.csv --val-path ./processing/validation/validation.csv --model-dir ./model --max-depth 6 --min-samples-leaf 2
python src/evaluate.py --model-dir ./model --test-path ./processing/test/test.csv --output-dir ./processing/eval
```

## Pipeline

```bash
python pipelines/pipeline.py --role arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole> --region us-west-2
```
