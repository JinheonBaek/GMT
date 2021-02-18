# HIV
python main.py  --type classification_OGB \
                --data ogbg-molhiv \
                --model GMT \
                --model-string GMPool_G-SelfAtt-GMPool_I \
                --gpu $1 \
                --experiment-number $2 \
                --batch-size 512 \
                --num-hidden 128 \
                --num-heads 4 \
                --lr-schedule \
                --cluster

# # Tox21
# python main.py  --type classification_OGB \
#                 --data ogbg-moltox21 \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 1 \
#                 --lr-schedule \
#                 --cluster

# # ToxCast
# python main.py  --type classification_OGB \
#                 --data ogbg-moltoxcast \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 8 \
#                 --lr-schedule \
#                 --cluster

# # BBBP
# python main.py  --type classification_OGB \
#                 --data ogbg-molbbbp \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 2 \
#                 --ln \
#                 --lr-schedule \
#                 --cluster
