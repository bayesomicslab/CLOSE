import argparse

def readArguments():
    """
    Read all command line arguments
    """

    parser = argparse.ArgumentParser(description="DEFAULT DESCRIPTION")
    subparsers = parser.add_subparsers(dest="command")

    parser_preprocessing = subparsers.add_parser("preprocess", help="Apply preprocessing to the specified data file")

    __build_parser_preprocessing(parser_preprocessing)


    args = parser.parse_args()

    __validate_preprocessing_args(args)

    return args

def __build_parser_preprocessing(parser: argparse.ArgumentParser):
    """
    Build the preprocessing parse for preprocessing arguments
    """

    parser.add_argument(
        "file",
        type=str,
        help="Path to the file to be processed"
    )
    parser.add_argument(
        "-extractEmbeddings",
        required=False,
        action="store_true",
        help="Extract the vector embeddings for a specific set of data, requires -saveFile and -model"
    )
    parser.add_argument(
        "-saveFile",
        required=False,
        type=str,
        help="Select which file will be saved to"
    )
    parser.add_argument(
        "-model",
        required=False,
        type=str,
        help="Select the model to be used for processing"
    )
    parser.add_argument(
        "-tokenizer",
        required=False,
        type=str,
        help="Select the tokenizer to be used for processing"
    )
    
def __validate_preprocessing_args(args):
    """
    Validate the preprocessing arguments passed in
    """

    pass
