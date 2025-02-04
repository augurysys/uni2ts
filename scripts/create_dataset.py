import json
import os

import pandas as pd

from collections.abc import Generator
from typing import Any

import datasets
from datasets import Features, Sequence, Value


base_dir = "data/SSS"

for dset in ['train', 'test', 'val']:
    source_dir = os.path.join(base_dir, dset)
    df = pd.read_parquet(os.path.join(source_dir, "extracted_features.parquet"))
    # x_windows = pd.read_parquet(os.path.join(source_dir, "x.parquet"))
    # y_windows = pd.read_parquet(os.path.join(source_dir, "y.parquet"))

    # ds_merged = pd.concat([x_windows, y_windows], axis=0)
    # ds_merged = ds_merged.sort_values(by=['window_id', 'timestamp'], axis=0)
    y_timestamps = pd.read_parquet(os.path.join(source_dir, "y_timestamps.parquet"))

    factory_metadata = pd.read_csv(os.path.join(source_dir, "factory-control-loop-tag.csv.xz"))
    with open(os.path.join(source_dir, "control-loop-config.json")) as f:
        control_loop_cfg = json.load(f)

    context_length = 60  # control_loop_cfg['preprocess']['tsWindowSize']
    prediction_length = control_loop_cfg['preprocess']['labelValidation']['labelsMatrixSizeInMinutes']

    tag_ids = factory_metadata[factory_metadata.tag_type == "DV"].tag_id.tolist() + factory_metadata[factory_metadata.tag_type == "MV"].tag_id.tolist()
    target_tag_ids = [str(c) for c in factory_metadata[factory_metadata.tag_type == "CV"].tag_id.tolist()]
    tag_ids = [str(c) for c in tag_ids]

    def multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
        for i, timestamp in y_timestamps['timestamp'].items():
            # if dset in ['val'] and i >= 100:
            #     break
            start_time = timestamp - pd.Timedelta(minutes=context_length)
            end_time = timestamp + pd.Timedelta(minutes=prediction_length)
            example = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
            if len(example) < (context_length + prediction_length):
                continue
            yield {
                "target": example[target_tag_ids].to_numpy().T,  # array of shape (var, time)
                "feat_dynamic_real": example[tag_ids].to_numpy().T,
                "start": example['timestamp'].iloc[0],
                "freq": 'min',
                "item_id": f"item_{i}",
            }

    features = Features(
        dict(
            target=Sequence(
                Sequence(Value("float32")), length=len(target_tag_ids)
            ),  # multivariate time series are saved as (var, time)
            feat_dynamic_real=Sequence(
                Sequence(Value("float32")), length=len(tag_ids)
            ),  # multivariate time series are saved as (var, time)
            start=Value("timestamp[s]"),
            freq=Value("string"),
            item_id=Value("string"),
        )
    )

    hf_dataset = datasets.Dataset.from_generator(
        multivar_example_gen_func, features=features
    )
    hf_dataset.save_to_disk(os.path.join(base_dir,f"dataset_dynamic_feat/{dset}_small"))

# ds_multi = datasets.load_from_disk(os.path.join(base_dir, "dataset/val")).with_format("numpy")
# print(ds_multi[0]["target"].shape)