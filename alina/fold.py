import sys
import os
import argparse
import torch
import naskit as nsk

from .alina import AliNA


    
def process(args):
    alina = AliNA.load(model="pretrained_augmented")
    
    device = torch.device(args.device)
    alina = alina.to(device)
    
    if args.mode=='seq':
        na = alina.fold(args.input, threshold=args.threshold)
        print(na.struct)
        return

    with nsk.dotRead(args.input) as f:
        nas = [na for na in f]

    nas = alina.fold(nas, threshold=args.threshold, batch_size=args.batch_size, verbose=True)
    
    with nsk.dotWrite(args.out) as w:
        for na in nas:
            w.write(na)
            
    print(f'\n{len(nas)} predictions were written to {args.out}')
    
    
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

    parser.add_argument('-bs', '--batch_size', type=int, default=8, metavar='<Batch size>',
                        help="Batch size for file mode. Default - 8.")
    
    parser.add_argument('-d', '--device', type=str, default="cpu", metavar='<Model device>', 
                        help='Pytorch device to run model. Default - cpu.')
    
    args = parser.parse_args()
    
    if args.out is None and args.mode=='file':
        path_comp = args.input.split(os.sep)
        path_comp[-1] = f'Prediction_{path_comp[-1]}'
        args.out = os.sep.join(path_comp)
        
    process(args)

    
    
    