import os
import argparse

#cap_audio_len=480000

def add_val(val):
    global cap_audio_len
    print(cap_audio_len)
    cap_audio_len+=val
    print(cap_audio_len)

def main(args):
    global cap_audio_len
    cap_audio_len=args.cap_len
    val=args.val

    add_val(val)

    print(cap_audio_len)

if __name__ == '__main__':
    #------------------------------------------------------------------------------
    # arguments
    #------------------------------------------------------------------------------
    parser = argparse.ArgumentParser (description = 'simple')
    parser.add_argument('--cap_len', type=int, help='Set the cap of audio length in samples (default 480000, equivalent to 30s) (OET/AC 240000)', default=480000)
    parser.add_argument('--val', type=int, help='Add to cap_len (def 1)', default=1)    
    args = parser.parse_args()
    main(args)
   
    
