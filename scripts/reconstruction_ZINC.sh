# ZINC
python main.py  --type reconstruction_ZINC \
                --data ZINC \
                --model GMT \
                --model-string GMPool_G \
                --gpu $1 \
                --experiment-number $2 \
                --batch-size 128 \
                --num-hidden 32 \
                --num-heads 1 \
                --cluster \
                