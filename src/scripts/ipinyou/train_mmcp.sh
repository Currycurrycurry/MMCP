python main.py --exp_kind mtl \
                --exp_name mmcp \
                --model MMCP \
                --dataset ipinyou \
                --camp $1 \
                --bid_prop 0.05 \
                --cdf_predict 0 \
                --z_size 301 \
                --win_lose_flag 0 \
                --train_bs 20480 \
                --predict_bs 20480 \
                --lr 0.00001 \
                --epoch 50 \
                --first_loss_weight 0.01\
                --second_loss_weight 0.1 \
                --B_START 0 \
                --B_LIMIT 300 \
                --B_DELTA 1
