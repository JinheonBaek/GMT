# grid
python main.py  --type reconstruction_synthetic \
                --data grid \
                --model GMT \
                --model-string GMPool_G \
                --gpu $1 \
                --experiment-number $2 \
                --num-hidden 32 \
                --num-heads 1 \
                --cluster \
                --ln

# ring
python main.py  --type reconstruction_synthetic \
                --data ring \
                --model GMT \
                --model-string GMPool_G \
                --gpu $1 \
                --experiment-number $2 \
                --num-hidden 32 \
                --num-heads 1 \
                --cluster \
                --ln
