clear

# 获取 Config().resume 的值
resume=$(python -c "from config import Config; print(Config().resume)")

# 如果 resume 为 False，则删除 'e_logs'、'ckpt'、'e_results' 文件夹
if [ "$resume" = "False" ]; then
    rm -rf e_logs ckpt e_results e_preds
fi

# 启动 Telegram 机器人监听
nohup python awake_lisa.py > bot/lisa_working.log 2>&1 &
echo "Lisa awake."


method=$(python -c "from config import Config; print(Config().task)")
devices=$(python -c "from config import Config; print(Config().device)")

# 如果没有设定默认值，可以在 Shell 脚本中手动指定
# Example: ./sub.sh tmp_proj 0,1,2,3 3 --> Use 0,1,2,3 for training, release GPUs, use GPU:3 for inference.
# method=${method:-"COD"}
# devices=${devices:-0}

# srun --nodes=1 --nodelist=Master,Slave1,Slave2,Slave3,Slave4,Slave5 \
# --ntasks-per-node=1 \
# --gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
# --cpus-per-task=32 \
bash train.sh ${method} ${devices}

hostname

devices_test=${3:-0}
bash test.sh ${devices_test}

python train_finish_notice.py
