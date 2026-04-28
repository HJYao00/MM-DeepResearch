index_file=e5_Flat.index
corpus_file=merged_reindexed_new.jsonl
retriever_name=e5
retriever_path=e5-base-v2

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu