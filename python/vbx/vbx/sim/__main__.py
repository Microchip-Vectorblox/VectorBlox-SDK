from .model import main as model_main
import sys


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--checksum', '-c', type=lambda x: int(x, 16))
    args = parser.parse_args()
    model_bytes = open(args.model_file, 'rb').read()
    sys.exit(model_main(model_bytes, args.checksum))


if __name__ == "__main__":
    main()
