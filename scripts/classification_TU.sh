# D&D
python main.py  --type classification_TU \
                --data DD \
                --model GMT \
                --model-string GMPool_G-SelfAtt-GMPool_I \
                --gpu $1 \
                --experiment-number $2 \
                --batch-size 10 \
                --num-hidden 32 \
                --num-heads 4 \
                --lr-schedule \
                --cluster

# # PROTEINS
# python main.py  --type classification_TU \
#                 --data PROTEINS \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 2 \
#                 --lr-schedule \
#                 --cluster

# # MUTAG
# python main.py  --type classification_TU \
#                 --data MUTAG \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 4 \
#                 --ln \
#                 --lr-schedule \
#                 --cluster

# # IMDB-B
# python main.py  --type classification_TU \
#                 --data IMDB-BINARY \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 1 \
#                 --lr-schedule \
#                 --cluster

# # IMDB-M
# python main.py  --type classification_TU \
#                 --data IMDB-MULTI \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 1 \
#                 --lr-schedule \
#                 --cluster

# # COLLAB
# python main.py  --type classification_TU \
#                 --data COLLAB \
#                 --model GMT \
#                 --model-string GMPool_G-SelfAtt-GMPool_I \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --num-heads 2 \
#                 --lr-schedule \
#                 --cluster