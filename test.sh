devices=${1:-0}
pred_root=${2:-e_preds}
log_file=./ckpt/$(python -c "from config import Config; print(Config().task)")/log.txt  # 将日志文件路径保存到一个变量中

# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root}

echo "Inference finished at $(date)" >> "${log_file}"  # 将推理完成时间写入日志文件

# Evaluation
log_dir=e_logs
mkdir ${log_dir}
testsets=CAMO_TestingDataset  && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=CHAMELEON_TestingDataset && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=COD10K_TestingDataset && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=NC4K_TestingDataset && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &

# Wait for evaluation tasks to finish
echo "Evaluating..." >> "${log_file}"  # 将开始评估时间写入日志文件
wait
echo "Evaluation finished at $(date)" >> "${log_file}"  # 将评估完成时间写入日志文件


python gen_best_ep.py