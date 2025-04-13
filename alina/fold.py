import sys
import os
import argparse
import naskit as nsk

from .alina import AliNA



def log(n, N):
    x = int(20*n/N)
    print(f"\r|{'-'*x}>{'.'*(20-x)}| {n} / {N}", end='')
    
    
def process(args):
    alina = AliNA.load(model="pretrained_augmented")
    if args.gpu:
        alina = alina.to("cuda")
    
    if args.mode=='seq':
        na = alina.fold(args.input, threshold=args.threshold)
        print(na.struct)
        return

    with nsk.dotRead(args.input) as f:
        nas = [na for na in f]

    with nsk.dotWrite(args.out) as w:
        N = len(nas)
        for n, na in enumerate(nas, start=1):
            na = alina.fold(na, threshold=args.threshold)
            w.write(na)
            log(n, N)
            
    print(f'\n{n} predictions were written to {args.out}')
    
    
def main():
    parser = argparse.ArgumentParser(description='AliNA args')
    
    parser.add_argument('-m', '--mode', type=str, 
                        choices=['seq', 'file'], default='seq', 
                        help='Prediction mode: "seq" - for single RNA sequence passed to command line. "file" - for multiple predictions from .fasta file.')
                        
    parser.add_argument('-i', '--input', type=str, 
                        required=True, metavar='<Sequence or Fasta file>', 
                        help='RNA sequence or path to the fasta file.')
                        
    parser.add_argument('-o', '--out', type=str, metavar='<Output file>',
                        help='Path to the output file for "file" mode. Default - Prediction_<input file name>.')
    
    parser.add_argument('-th', '--threshold', type=float, default=0.5, metavar='<Threshold value>',
                        help="Threshold value in range [0, 1] for output processing. The bigger value gives less complementary bonds. Default - 0.5")
    
    parser.add_argument('--gpu', default=False, action='store_true', 
                        help='Run model on gpu if CUDA is available. Default - run on cpu.')
    
    parser.add_argument('--skip-errors', default=False, action='store_true', 
                        help='Skip invalid sequences instead of raising SequenceError. Default - False.')
    
    parser.add_argument('--no-warn', default=False, action='store_true', 
                        help='Do not raise warning on invalid sequnce skip. Default - False.')
    
    args = parser.parse_args()
    
    if args.out is None and args.mode=='file':
        path_comp = args.input.split(os.sep)
        path_comp[-1] = f'Prediction_{path_comp[-1]}'
        args.out = os.sep.join(path_comp)
        
    process(args)

    
    
    