from datasets import load_from_disk

def main(args):
    test_data = args.TEST_DATA #'/scratches/dialfs/alta/sb2549/wav2vec2_exp/data_vectors_attention/LIESTcal01/LIESTcal01_part4_att.hf/'
    test_dataset = load_from_disk(test_data)
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    test_ds = test_dataset.map(predict, batched=True, batch_size=32)                                                                                                                  

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TEST_DATA', type=str, help='file with CAV')
    commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='file to save prediction')
    args = commandLineParser.parse_args()
    main(args)
