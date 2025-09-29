from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

schema = CollectionSchema(
    name="Sampling_Ceval",
    datasets=[
        DatasetInfo(
            name="ceval",
            weight=1,
            task_type="exam",
            tags=["zh"],
            args={"few_shot_num": 0},
        )
    ],
)

# get the mixed data
mixed_data = WeightedSampler(schema).sample(
    100
)  # set a large number to ensure all datasets are sampled
# dump the mixed data to a jsonl file
dump_jsonl_data(mixed_data, "sampling_ceval_test.jsonl")
