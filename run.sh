data_dir="./data/input/"
# bert_model="bert-base-chinese"
bert_model="./data/model/bert-base-chinese-pytorch_model.bin"
config_file="./data/model/bert-base-chinese-config.json"
task_name="xgfy"
vacab_root="./data/model/"
output_dir="./data/output/"
cache_dir="./data/model/"

weight_name="net_weight_10.bin"
config_name="config_name_10.bin"

max_seq_length=100

learning_rate=5e-5
num_train_epochs=10
train_batch_size=64
eval_batch_size=8
log_frq=40

gradient_accumulation_steps=1
n_warmup=2000
local_rank=-1
seed=42
loss_scale=0

do_train="do_train"
do_eval=""
do_lower_case="do_lower_case"
no_cuda=""
fp16=""

python train.py --data_dir $data_dir --bert_model $bert_model --task_name $task_name --vacab_root $vacab_root --output_dir $output_dir --config_file $config_file --cache_dir $cache_dir --max_seq_length $max_seq_length --log_frq $log_frq --train_batch_size $train_batch_size --num_train_epochs $num_train_epochs --eval_batch_size $eval_batch_size --learning_rate $learning_rate --n_warmup $n_warmup --gradient_accumulation_steps $gradient_accumulation_steps --local_rank $local_rank --seed $seed --loss_scale $local_rank --weight_name $weight_name --config_name $config_name --do_lower_case --do_eval