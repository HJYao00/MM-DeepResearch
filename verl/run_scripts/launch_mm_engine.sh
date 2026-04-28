index_file=jina-clip-v2_Flat_image.index
corpus_file=total_image_search_result_rag.parquet
retriever_name=jina-clip-v2
retriever_path=jina-embeddings-v4

echo ${index_file}
echo ${corpus_file}

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server_image.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu