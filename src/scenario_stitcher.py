import argparse
import json
import os



def get_json_iteration(filename: str) -> int:
    return int(filename[filename.index('_')+1:-5].replace('agents_', ''))

def main():

    # Process the params
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The path(s) to the folder(s) containing all of the generated data', required=True)
    parser.add_argument('-o', '--output', help='The path to write all of the processed data to.',
                        required=True)
    parser.add_argument('-s', '--suffix', help='Suffix for written file.',
                        required=True)
    parser.add_argument('-v', '--verbose', help='Will print out informative information to the terminal.',
                        action='store_true')

    parser = parser.parse_args()

    if not os.path.isdir(parser.path):
        print('Please make sure the path specified is a directory...')
        return

    json_snapshots = {}

    for root, _, files in os.walk(parser.path, topdown=True):

        json_files = [f for f in files if f[-4:] == 'json']

        print("Found %s simulations..." % str(len(json_files)))

        json_files.sort(key=get_json_iteration)

        for file in json_files:
            with open(os.path.join(root, file)) as json_file:

                key = get_json_iteration(file)
                json_snapshots[key] = json.load(json_file)

    print('Writing data to output file:' + parser.output + '/processed_agents_' + parser.suffix + '.json:')
    with open(parser.output + '/processed_agents_' + parser.suffix + '.json', 'w') as outfile:
        json.dump(json_snapshots, outfile, indent=4)


if __name__ == '__main__':
    main()