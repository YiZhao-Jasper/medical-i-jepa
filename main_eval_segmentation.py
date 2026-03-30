import argparse
import yaml
from src.evaluate_segmentation import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str, required=True,
    help='path to eval config yaml')


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.fname, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    eval_main(params)
