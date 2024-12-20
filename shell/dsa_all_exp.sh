#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3;

# python main_tune.py --data dailysports --encoder CNN  --agent AIR --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent AIR --norm LN & wait;\

python main_tune.py --data dailysports --encoder CNN  --agent SFT --norm BN & wait;\

python main_tune.py --data dailysports --encoder CNN  --agent SFT --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent Offline --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent Offline --norm LN & wait;\

# # Regularization
# python main_tune.py --data dailysports --encoder CNN  --agent LwF --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent LwF --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent MAS --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent MAS --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent DT2W --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent DT2W --norm LN & wait;\

# # Replay
# python main_tune.py --data dailysports --encoder CNN  --agent ER --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent ER --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent DER --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent DER --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent Herding --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent Herding --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent ASER --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent ASER --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent CLOPS --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent CLOPS --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent FastICARL --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent FastICARL --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent GR --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent GR --norm LN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent ER_Sub --norm BN & wait;\

# python main_tune.py --data dailysports --encoder CNN  --agent ER_Sub --norm LN & wait;\

#python main_tune.py --data dailysports --encoder CNN  --agent Inversion --norm BN & wait;\
#
#python main_tune.py --data dailysports --encoder CNN  --agent Inversion --norm LN & wait;\
#
#python main_tune.py --data dailysports --encoder CNN  --agent Mnemonics --norm BN & wait;\
#
#python main_tune.py --data dailysports --encoder CNN  --agent Mnemonics --norm LN & wait;\
