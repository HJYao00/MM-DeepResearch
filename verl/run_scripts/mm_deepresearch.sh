mkdir -p logs_dir/logs_mm_deepresearch_8b

function now() {
    date '+%Y-%m-%d-%H-%M'
}

export JUDGE_API_BASE="http://xxx:8001/v1"
export JUDGE_API_KEY="EMPTY"
export JUDGE_MODEL_NAME="judge_model"

export SEARCH_CACHED_DATA_PATHS=lens_cached_data.output_with_cached_image_path.jsonl

VAL_DATA_1=test_1.parquet
VAL_DATA_2=test_2.parquet
VAL_DATA_4=test_3.parquet

TRAIN_DATA_1=train_data.parquet

NAME=mm-deepresearch-8b

nohup bash examples/sglang_multiturn/search_r1_like/run_qwen3vl-8b_instruct_search_multiturn_sglang_mmdeepresearch.sh \
  trainer.experiment_name=${NAME}-$(now) \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
  actor_rollout_ref.model.path=your_model_path \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/sglang_multiturn/config/tool_config/search_tool_config_final_model.yaml \
  data.train_files=[${TRAIN_DATA_1}] \
  data.val_files=[${VAL_DATA_1},${VAL_DATA_2},${VAL_DATA_4}]  \
  trainer.default_local_dir=./mm_deepresearch/${NAME}-$(now) \
  trainer.n_gpus_per_node=8 \
  > logs_dir/logs_mm_deepresearch_8b/${NAME}-$(now).log 2>&1 &
