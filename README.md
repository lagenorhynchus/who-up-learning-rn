# who-up-learning-rn
CSCI1470 Deep Learning final project 2026: Liam Capozza, Janet Joseph, Siddhant Karmali

Training
1. Backprop sanity check (run this first)
Overfits a single batch of 16 images for 100 epochs. Loss must approach 0.

python train.py --overfit
To run longer or with a higher learning rate:

python train.py --overfit --overfit_epochs 200 --lr 1e-2
2. Development run (10-class subset, CPU/MPS)
python train.py --epochs 20 --batch_size 64 --lr 1e-3
3. Full run (OSCAR cluster, 100 classes)
python train.py --epochs 50 --batch_size 128 --num_classes 100 \
                --data_dir /path/to/data --save_dir ./outputs --num_workers 8
                