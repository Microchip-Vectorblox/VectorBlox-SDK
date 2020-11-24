import argparse
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    args = parser.parse_args()

    images = [os.path.basename(x).split('.')[0] for x in sorted(glob.glob(os.path.join(args.folder, '*.jpg')))]
    with open('images.txt', 'w') as f:
        for image in images:
            f.write('{}\n'.format(image))
