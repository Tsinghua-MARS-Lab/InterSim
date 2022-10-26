# Usage
# python scripts/combine_pickle_files.py -i PICKLE_DATA_PATH1 PICKLE_DATA_PATH2 ...

import argparse
import numpy as np
import tqdm

import pickle5 as pickle

def combine_pickle(args):
    combined_dict = {}

    for fname in args.input_paths:
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        for key in tqdm.tqdm(data):
            if key not in combined_dict:
                combined_dict[key] = data[key]
            else:
                existing_data = combined_dict[key]
                extra_data = data[key]
                # Combining results from multiple files.
                combined_data = {
                    'ids': existing_data["ids"] + extra_data["ids"],
                    'score': np.concatenate([existing_data['score'], extra_data['score']], 0),
                    'rst': np.concatenate([existing_data['rst'], extra_data['rst']], 0)
                }

                assert len(combined_data["ids"]) == 2, "data id {}, data len {}".format(key, len(combined_data["ids"]))

                combined_dict[key] = combined_data

    with open(args.output_path, 'wb') as f:
        pickle.dump(combined_dict, f, pickle.HIGHEST_PROTOCOL)

    print('Combining {} files into {} examples at {}'.format(len(args.input_paths), len(combined_dict), args.output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input-paths', nargs='+', help='List of input pickle file to read')
    parser.add_argument('-o', '--output-path', help='Path of output pickle file to save')
    args = parser.parse_args()

    combine_pickle(args)


if __name__ == '__main__':
    main()
