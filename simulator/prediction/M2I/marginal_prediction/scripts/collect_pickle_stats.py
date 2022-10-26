# Usage
# python scripts/collect_pickle_states.py -i PICKLE_DATA_PATH

import argparse
import numpy as np
import tqdm

import structs


def collect_stats(args):
    data = structs.load(args.input_path)

    sample_count = []
    for key in tqdm.tqdm(data):
        sample_count.append(data[key]['rst'].shape[0])

    values, counts = np.unique(sample_count, return_counts=True)

    value_counts_pairs = list(zip(values, counts))
    for (value, count) in value_counts_pairs:
        print('{} has {} predictions with counts {}'.format(args.input_path, value, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input-path', help='Input pickle file to read')
    args = parser.parse_args()

    collect_stats(args)


if __name__ == '__main__':
    main()
