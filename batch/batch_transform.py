import argparse
import os
from sagemaker import Session
from sagemaker.transformer import Transformer
import boto3

def run(model_data_s3_uri: str, input_s3_uri: str, output_s3_uri: str, region: str = "us-west-2", instance_type: str = "ml.m5.large"):
    sm = Session(boto3.Session(region_name=region))
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    transformer = Transformer(
        model_name=None,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_s3_uri,
        sagemaker_session=sm,
        image_uri=image_uri,
        model_data=model_data_s3_uri,
    )
    transformer.transform(data=input_s3_uri, content_type="text/csv", split_type="Line")
    transformer.wait()
    print(f"Batch output at: {output_s3_uri}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-data", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    p.add_argument("--instance-type", default="ml.m5.large")
    a = p.parse_args()
    run(a.model_data, a.input, a.output, a.region, a.instance_type)
