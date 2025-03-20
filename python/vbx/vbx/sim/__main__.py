from .model import main as model_main
import sys


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--checksum', '-c', type=lambda x: int(x, 16))
    parser.add_argument('--debug', '-d', action='store_true')
    args = parser.parse_args()
    sys.exit(model_main(args.model_file, args.checksum, args.debug))


if __name__ == "__main__":
    main()
