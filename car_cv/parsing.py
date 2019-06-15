import json
import os
import argparse
import builtins


def evaluate_type(type_name: str) -> type:
    """
    Converts a string representing a builtin type to the actual type object in a safe manner.

    Args:
        type_name: A string representing a builtin type.

    Returns:
        The corresponding type object.
    """
    type_obj = getattr(builtins, type_name, None)
    if type_obj is not None and isinstance(type_obj, type):
        return type_obj

    else:
        raise ValueError(f'Unknown type {type_name}.')


def make_parser(argument_spec_path: str) -> argparse.ArgumentParser:
    """
    Instantiates an `argparse.ArgumentParser` object from a JSON-formatted file that decodes to a `list` of `dicts`,
    each of which represents a single argument and contains two keys: `command`, which should correspond  to a `list`
    containing the names of the argument, and `options`, which should correspond to a (possibly empty) `dict` containing
    keyword arguments to pass to `parser.add_arguments`.

    Args:
        argument_spec_path: The path to the JSON-formatted file containing the argument specification.

    Returns:
        An `argparse.ArgumentParser` instantiated based on the provided specification.

    Notes:
        This method of instantiating parsers is more restrictive than doing so manually. In particular, only types from
        __builtin__ (int, str, etc.) are supported for the `type` optional keyword argument, and extended functionality
        such as subparsers and mutually exclusive arguments are not available.
    """
    parser = argparse.ArgumentParser()

    with open(os.path.join(argument_spec_path)) as f:
        arguments = json.load(f)
        for argument in arguments:
            if 'type' in argument['options']:
                argument['options']['type'] = evaluate_type(argument['options']['type'])
            parser.add_argument(*argument['command'], **argument['options'])

    return parser
