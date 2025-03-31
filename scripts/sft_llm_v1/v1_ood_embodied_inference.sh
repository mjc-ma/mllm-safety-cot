test_dataset="ood"
ood_test_scene="embodied"
num_gpus=4 # 使用的 GPU 数量
world_size=4  # 任务总数量/数据集切分块总数
batch_size=4
model_name="sft_cot_llama+sharegpt4v01" # v1模型
checkpoint=2200
num_examples=all

for rank in $(seq 0 $((world_size - 1))); do
    gpu_id=$((rank % num_gpus))  # 根据 rank 分配 GPU 编号
    echo "Launching task with rank ${rank} on GPU ${gpu_id}..."
    save_path="./logs/sft_answer/model:${model_name}_${checkpoint}/test:${test_dataset}_samples:${num_examples}/scene:${ood_test_scene}/$(printf "%05d" ${rank})-$(printf "%05d" ${world_size}).json"

    CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. python src/inference.py \
    --model_path "/fs-computility/ai-shen/majiachen/results/${model_name}/checkpoint-${checkpoint}" \
    --save_log_path ${save_path} \
    --num_examples ${num_examples} \
    --test_dataset ${test_dataset} \
    --cot True \
    --world_size ${world_size} \
    --rank ${rank} \
    --batch_size ${batch_size} &
done

wait  # 等待所有脚本完成
