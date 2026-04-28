import requests
import pdb

retrieval_service_url = "http://127.0.0.1:8001/retrieve"

payload = {"queries": ['Google I/O 2024 Keynote: Sundar Pichai opening remarks\n', 'elon mask\n'], "topk": 3, "return_scores": True}

headers = {"Content-Type": "application/json", "Accept": "application/json"}

response = requests.post(
        retrieval_service_url,
        headers=headers,
        json=payload,
        timeout=30,
    )

pdb.set_trace()