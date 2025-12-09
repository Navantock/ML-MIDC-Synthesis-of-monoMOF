from argparse import ArgumentParser
import yaml


def read_args_from_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default="./configs/template.yaml")
    args = parser.parse_args()

    args = read_args_from_yaml(args.config_path)

    dataset_args = read_args_from_yaml(args["dataset_config_path"])
    model_args = read_args_from_yaml(args["model_config_path"]) if args.get("model_config_path", None) is not None else None
    train_args = read_args_from_yaml(args["train_config_path"]) if args.get("train_config_path", None) is not None else None
    return args, dataset_args, model_args, train_args