import argparse
import os
from sagemaker.model import Model
from sagemaker import Session
import boto3

def deploy(model_data_s3_uri: str, role: str, region: str = "us-west-2", endpoint_name: str = "iris-mlops-endpoint"):
    sm = Session(boto3.Session(region_name=region))
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    model = Model(image_uri=image_uri, model_data=model_data_s3_uri, role=role, sagemaker_session=sm)
    model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=endpoint_name)
    print(f"Deployed endpoint: {endpoint_name}")
    return endpoint_name

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-data", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    p.add_argument("--endpoint-name", default="iris-mlops-endpoint")
    a = p.parse_args()
    deploy(a.model_data, a.role, a.region, a.endpoint_name)
