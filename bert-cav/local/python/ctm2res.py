#!/usr/bin/env python3

import argparse
import os
import sys


def get_response(ctm_file, re_file):
    response = []
    current_file_name = None
    current_sentence = []

    with open(ctm_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split()
            part = parts[0].split('_')
            file_name = part[0]
            word = parts[-2]

            # If it's a new file, append the previous sentence (if any)
            if current_file_name is None:
                current_file_name = file_name
            elif current_file_name != file_name:
                response.append(current_file_name + " " + " ".join(current_sentence))
                current_sentence = []
                current_file_name = file_name

            # Append the word to the current sentence (skip 'sil' and 'sp')
            if word not in ['sil', 'sp']:
                current_sentence.append(word)

        # Append the last sentence (if any)
        if current_sentence:
            response.append(current_file_name + " " + " ".join(current_sentence))

    with open(re_file, 'w') as outfile:
        outfile.write("\n".join(response))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("ctm_file", help="input ctm")
    parser.add_argument("re_file", help="output reponse")
    args = parser.parse_args()

   # Save the command line input
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/mlf2hyp.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    get_response(args.ctm_file, args.re_file)

if __name__ == "__main__":
    main()


