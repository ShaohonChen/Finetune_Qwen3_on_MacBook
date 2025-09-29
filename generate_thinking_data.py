import os
import json
import argparse
from openai import OpenAI


# power by @不要葱姜蒜
SYSTEM_PROMPT = "你是由{zh_author}基于Transformer架构研发的大语言模型。你的名字叫{zh_name}，是一个人工智能助手。You are a large language model developed by the {en_author} based on the Transformer architecture. Your name is {en_name} and you are an artificial intelligence assistant."


def generate_response_from_api(prompt, think=False, system_prompt=None, key="EMPTY"):
    global client
    if "client" not in globals():
        client = OpenAI(
            api_key=key,
            base_url="https://api.siliconflow.cn/v1",
        )
    messages = []
    if system_prompt is not None:
        messages += [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
    messages += [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.1",
        messages=messages,
        extra_body={"enable_thinking": think},
    )
    response_text = completion.choices[0].message.content
    reasoning_content = completion.choices[0].message.reasoning_content if think else ""
    return response_text, reasoning_content


def main(
    key="self-cognition/self_cognition.jsonl",
    name="小鹅",
    author="SwanLab团队",
    eng_name="XiaoE",
    eng_author="SwanLab Team",
):
    with open("self-cognition/self_cognition.jsonl", "r") as fread:
        data_list = fread.readlines()

    generate_datas = []
    generate_datas_with_chatml = []
    for i, data in enumerate(data_list):
        if i == 5:  # for debug
            break
        print(f"================== 数据合成进展 ({i+1}/{len(data_list)})")
        data = json.loads(data)
        user_text = data["query"]
        print("=== 用户问题 ===")
        print(user_text)
        response_text, thinking_content = generate_response_from_api(
            user_text,
            think=True,
            system_prompt=SYSTEM_PROMPT.format(
                zh_name=name,
                zh_author=author,
                en_name=eng_name,
                en_author=eng_author,
            ),
            key=key,
        )
        print("=== 思考过程 ===")
        print(thinking_content)
        print("\n=== 回复内容 ===")
        print(response_text)

        generate_datas.append(
            {
                "query": user_text,
                "assistant": f"<think>\n{thinking_content}\n</think>\n\n{response_text}",
                "tag": data["tag"],
            }
        )
        generate_datas_with_chatml.append(
            {
                "messages": [
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": response_text,
                        "reasoning_content": thinking_content,
                    },
                ],
                "tag": data["tag"],
            }
        )

    # 写出数据
    os.makedirs("./self_cognition_with_thinking", exist_ok=True)
    with open(
        "./self_cognition_with_thinking/self_cognition_with_thinking.jsonl",
        "w",
        encoding="utf-8",
    ) as fwrite:
        for data in generate_datas:
            fwrite.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一个简单的脚本，接受 name 和 author 参数。"
    )
    parser.add_argument("--key", type=str, required=True, help="使用的API Key")
    parser.add_argument("--name", type=str, default="小鹅", help="指定数据集中模型名称")
    parser.add_argument(
        "--author", type=str, default="情感机器实验室", help="指定数据集中模型作者名称"
    )
    parser.add_argument(
        "--en_name", type=str, default="XiaoE", help="指定数据集中模型英文名称"
    )
    parser.add_argument(
        "--en_author",
        type=str,
        default="Emotion Machine Lab",
        help="指定数据集中模型作者英文名称",
    )
    args = parser.parse_args()

    main(args.key, args.name, args.author, args.en_name, args.en_author)
