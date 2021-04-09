import argparse
from run import run
import os,logging
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__ == "__main__":
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Train the model', action="store_true")
    parser.add_argument('--test', help='Run tests on the model', action="store_true")
    parser.add_argument('--export', help='Export the model as .pb', action="store_true")
    parser.add_argument('--small', help='Run FSRCNN-small', action="store_true")
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=3)
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs during training', default=20)

    parser.add_argument('--d', type=int, help='Variable for d', default=56)
    parser.add_argument('--s', type=int, help='Variable for s', default=12)
    parser.add_argument('--m', type=int, help='Variable for m', default=4)

    parser.add_argument('--traindir', help='Path to train images')
    parser.add_argument('--validdir', help='Path to validation images')

    args = parser.parse_args()
    params = [args.d,args.s,args.m]

    if args.small:
        params = [32,5,1]
    with tf.device('/device:GPU:2'):
        run = run(args.scale,args.batch,args.epochs,params,args.validdir)
        if args.train:
            run.train(args.traindir)
        elif args.test:
            run.test()
        elif args.export:
            pass