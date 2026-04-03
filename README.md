#training
python main.py --mode train --train-epochs 2 --train-lr 5e-4 --dice-lambda 0.5

#inferring 
python main.py --mode infer --train-epochs 50 --train-lr 1e-3 --dice-lambda 0.5 --infer-max-samples 10

# choose flags based on usage, all flags implemented in main.py
