import json
import os
from argparse import ArgumentParser


def make_parser(argument_spec_path: str) -> ArgumentParser:
    parser = ArgumentParser()

    with open(os.path.join(argument_spec_path)) as f:
        arguments = json.load(f)
        for argument in arguments:
            parser.add_argument(*argument['command'], **argument['options'])

    return parser
