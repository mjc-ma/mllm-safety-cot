test_dataset="general"
num_gpus=8 # 使用的 GPU 数量
world_size=8  # 任务总数量/数据集切分块总数
batch_size=4
model_name="sft_cot_llama+sharegpt4v01" # v1模型
checkpoint=2200
num_examples=500

############################## 用finetune过的模型推理，保存推理结果 ##############################

# for rank in $(seq 0 $((world_size - 1))); do
#     gpu_id=$((rank % num_gpus))  # 根据 rank 分配 GPU 编号
#     echo "Launching task with rank ${rank} on GPU ${gpu_id}..."
    
#     CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. python src/inference.py \
#     --model_path "/fs-computility/ai-shen/majiachen/results/${model_name}/checkpoint-${checkpoint}" \
#     --save_log_path "./logs/sft_answer/model:${model_name}_${checkpoint}/test:${test_dataset}_samples:${num_examples}/$(printf "%05d" ${rank})-$(printf "%05d" ${world_size}).json" \
#     --num_examples ${num_examples} \
#     --test_dataset ${test_dataset} \
#     --cot True \
#     --world_size ${world_size} \
#     --rank ${rank} \
#     --batch_size ${batch_size} &
# done
# wait  # 等待所有脚本完成

############################## 用score模型给判断是否安全（危险问题拒答、非危险问题做出回答） ##############################
scores=()  # 存放所有得分的数组
output_dir="./logs/sft_answer/model:${model_name}_${checkpoint}/test:${test_dataset}_samples:${num_examples}"  # 结果存放目录
mkdir -p "${output_dir}"
tmp_score_file="${output_dir}/tmp_scores.txt"
> "$tmp_score_file"

for rank in $(seq 0 $((world_size - 1))); do
    gpu_id=$((rank % num_gpus))  # 根据 rank 分配 GPU
    file="${output_dir}/$(printf "%05d" ${rank})-$(printf "%05d" ${world_size}).json"
    (
        echo "Evaluating files with rank ${rank} on GPU ${gpu_id}..."
        score=$(CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. python src/eval.py \
            --model_path "/fs-computility/ai-shen/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct" \
            --input_path ${file} \
            --save_score_path ${file} \
            --batch_size ${batch_size} | tail -n 1)
        if [[ "$score" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "$score" >> "$tmp_score_file"  # 仅写入数值
            echo "Rank ${rank} score: ${score}" 
        else
            echo "Rank ${rank} returned an invalid score: ${score}"
        fi
    ) &
done
wait  

################################ 计算平均得分 ####################################
total=0
count=0
while IFS= read -r s; do
    if [[ "$s" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then  # 确保是数值
        total=$(awk "BEGIN {print $total + $s}")  
        count=$((count + 1))
    fi
done < "$tmp_score_file"
if [[ "$count" -gt 0 ]]; then
    avg_score=$(awk "BEGIN {print $total / $count}")
else
    avg_score="N/A (No valid scores)"
fi
final_score_file="${output_dir}/final_scores.txt"
echo "Final average score: ${avg_score}" | tee -a "$final_score_file"
rm -f "$tmp_score_file"
echo "All scores and final average score saved to ${final_score_file}"