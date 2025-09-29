import sys
from evalscope import TaskConfig, run_task


task_cfg = TaskConfig(
    model=sys.argv[1] if len(sys.argv) > 2 else "Qwen3-0.6B",  # 换成对应的模型
    api_url="http://127.0.0.1:8080/v1/chat/completions",
    eval_type="openai_api",
    datasets=[
        "data_collection",
    ],
    dataset_args={
        "data_collection": {
            "dataset_id": "sampling_ceval_test.jsonl",
            "filters": {"remove_until": "</think>"},  # 过滤掉思考的内容
        }
    },
    eval_batch_size=1,
    generation_config={
        "max_tokens": 2000,  # 最大生成token数，设置为2000，不然太慢了
        "temperature": 0.0,  # 采样温度
        "top_p": 0.95,  # top-p采样
        "top_k": 1,  # top-k采样
        "n": 1,  # 每个请求产生的回复数量
    },
    timeout=60000,  # 超时时间
    stream=False,  # 是否使用流式输出
    limit=100,  # 设置为100条数据进行测试
)

run_task(task_cfg=task_cfg)
