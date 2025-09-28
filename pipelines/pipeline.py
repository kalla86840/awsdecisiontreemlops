# SageMaker Pipelines: preprocess -> train (DecisionTree) -> evaluate -> register
import os
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker import Session
import boto3

def get_pipeline(region: str, role: str, default_bucket: str = None, base_job_prefix: str = "iris-mlops"):
    sagemaker_session = Session(boto3.Session(region_name=region))
    default_bucket = default_bucket or sagemaker_session.default_bucket()
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    instance_type = ParameterString("InstanceType", default_value="ml.m5.large")
    max_depth = ParameterInteger("MaxDepth", default_value=0)
    min_samples_leaf = ParameterInteger("MinSamplesLeaf", default_value=1)
    dataset_name = ParameterString("DatasetName", default_value="iris")

    sklearn_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

    # Processing: preprocess
    script_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
    )

    step_process = ProcessingStep(
        name="PreprocessData",
        processor=script_processor,
        code="src/preprocess.py",
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        job_arguments=[
            "--output-base", "/opt/ml/processing",
            "--dataset", dataset_name,
            "--cars-csv", "data/cars.csv",
        ],
        cache_config=cache_config,
    )

    # Training (DecisionTree)
    sklearn_estimator = SKLearn(
        entry_point="src/train.py",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "max-depth": max_depth,
            "min-samples-leaf": min_samples_leaf,
        },
    )

    train_input = TrainingInput(
        step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        content_type="text/csv",
    )
    val_input = TrainingInput(
        step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
        content_type="text/csv",
    )

    step_train = TrainingStep(
        name="TrainDecisionTree",
        estimator=sklearn_estimator,
        inputs={"train": train_input, "validation": val_input},
        cache_config=cache_config,
    )

    # Evaluation
    eval_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
                input_name="test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    # Register
    model = Model(
        image_uri=sklearn_image,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=sagemaker_session,
    )

    step_register = ModelStep(
        name="RegisterModel",
        model=model,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        register=True,
        model_package_group_name="iris-mlops-registry",
    )

    pipeline = Pipeline(
        name="IrisMLOpsPipeline",
        parameters=[instance_type, max_depth, min_samples_leaf, dataset_name],
        steps=[step_process, step_train, step_eval, step_register],
        sagemaker_session=sagemaker_session,
    )
    return pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--role", required=True)
    a = parser.parse_args()
    p = get_pipeline(region=a.region, role=a.role)
    print(p.upsert(role_arn=a.role))
