import os
import json
import argparse


def main(
    data_path="self-cognition/self_cognition.jsonl",
    output_dir="/mlx_data",
    name="小鹅",
    author="SwanLab团队",
    eng_name="Little-Swan",
    eng_author="SwanLab Team",
):
    mlx_data = []

    with open(data_path, "r") as fread:
        data_list = fread.readlines()

        for data in data_list:
            data = json.loads(data)
            user_text = data["query"]
            if data["tag"] == "zh":
                assistant_text = (
                    data["response"]
                    .replace("{{NAME}}", name)
                    .replace("{{AUTHOR}}", author)
                )
            else:
                assistant_text = (
                    data["response"]
                    .replace("{{NAME}}", eng_name)
                    .replace("{{AUTHOR}}", eng_author)
                )
            mlx_data.append(
                {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
            )

    # splite data
    val_data_num = len(mlx_data) // 5
    mlx_train_data = mlx_data[val_data_num:]
    mlx_val_data = mlx_data[:val_data_num]

    # write data
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as fwrite:
        for data in mlx_train_data:
            fwrite.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "valid.jsonl"), "w", encoding="utf-8") as fwrite:
        for data in mlx_val_data:
            fwrite.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一个简单的脚本，接受 name 和 author 参数。"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="指定读取数据集文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mlx_data",
        help="指定写出数据集文件夹路径",
    )
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

    main(
        args.data_path,
        args.output_dir,
        args.name,
        args.author,
        args.en_name,
        args.en_author,
    )
