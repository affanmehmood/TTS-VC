import sys
import argparse
import os

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Multiple models.')
    parser.add_argument('-input',
                        type=str,
                        help='Input folder path e.g. path/path/', required=True)
    parser.add_argument('-output',
                        type=str,
                        help='Output folder path e.g. path/path/', required=True)

    args = parser.parse_args()

    sys.argv = ["filename", "-input", args.input, "-output", args.output]

    print("python VC.py " + " ".join(sys.argv))

    os.system("python VC.py " + " ".join(sys.argv[1:]))
