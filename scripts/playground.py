import copy
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
SIZE = "base"  # model size: choose from {'small', 'base', 'large'}
PDT = 30  # prediction length: any positive integer
CTX = 120  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 6  # batch size: any positive integer
# TEST = 1000  # test set length: any positive integer

source_dir = "../data/BazanSplitter/test"
checkpoint_path = "../outputs/finetune/moirai_1.1_R_base/BazanSplitter_train/BazanSplitter_finetune/checkpoints/epoch=0-step=1000.ckpt"

df = pd.read_parquet(os.path.join(source_dir, "extracted_features.parquet"))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').head(15000)

TEST = len(df)
# df = df.set_index("timestamp").sort_index()
# df = df.head(10000)
# Convert into GluonTS dataset
# ds = PandasDataset(dict(df), freq="1min", unchecked=True, target='228013821')

factory_cfg = pd.read_csv(os.path.join(source_dir, "factory-control-loop-tag.csv.xz"))
cv_tags = [str(c) for c in factory_cfg[factory_cfg.tag_type == "CV"].tag_id.unique()]
mv_dv_tags = [str(c) for c in factory_cfg[factory_cfg.tag_type != "CV"].tag_id.unique()]

ds = PandasDataset(
    df,
    target=cv_tags,
    timestamp="timestamp",
    feat_dynamic_real=mv_dv_tags,
    freq="1min",
    unchecked=True
)

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST-PDT+1,  # number of windows in rolling window evaluation
    distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
)

# Prepare pre-trained model by downloading model weights from hugg1ingface hub
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=len(cv_tags),
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    finetuned_model = MoiraiForecast.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        prediction_length=PDT,
        target_dim=len(cv_tags),
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        context_length=CTX
    )

    # finetuned_model = MoiraiForecast(
    #     module=MoiraiModule.from_pretrained('/home/dbarsky/Code/uni2ts/weights/BazanSplitter'),
    #     prediction_length=PDT,
    #     context_length=CTX,
    #     patch_size=PSZ,
    #     num_samples=100,
    #     target_dim=len(cv_tags),
    #     feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    #     past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    # )
elif MODEL == "moirai-moe":
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=16,
        num_samples=100,
        target_dim=len(cv_tags),
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

predictor = model.create_predictor(batch_size=BSZ)
finetuned_predictor = finetuned_model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)
finetuned_forecasts = finetuned_predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)
finetuned_forecast_it = iter(finetuned_forecasts)

idx = 0
predictions = []
finetuned_predictions = []
while True:
    try:
        inp = next(input_it)
        label = next(label_it)
        forecast = next(forecast_it)
        finetuned_forecast = next(finetuned_forecast_it)
    except StopIteration:
        break
    for lookahead in range(PDT):
        prediction = {
            "timestamp": label['start'].start_time + pd.Timedelta(minutes=lookahead),
            "lookahead": lookahead + 1,
        }
        for i, tag_id in enumerate(cv_tags):
            prediction[f'true_{tag_id}'] = label['target'][i, lookahead]
            prediction[f'pred_{tag_id}'] = np.median(forecast.samples[:, lookahead, i])
            prediction[f'prctile_05_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 0.05)
            prediction[f'prctile_95_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 0.95)
            prediction[f'prctile_25_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 0.25)
            prediction[f'prctile_75_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 0.75)
        predictions.append(prediction)
        finetuned_prediction = copy.deepcopy(prediction)
        for i, tag_id in enumerate(cv_tags):
            finetuned_prediction[f'true_{tag_id}'] = label['target'][i, lookahead]
            finetuned_prediction[f'pred_{tag_id}'] = np.median(finetuned_forecast.samples[:, lookahead, i])
            finetuned_prediction[f'prctile_05_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 0.05)
            finetuned_prediction[f'prctile_95_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 0.95)
            finetuned_prediction[f'prctile_25_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 0.25)
            finetuned_prediction[f'prctile_75_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 0.75)
        finetuned_predictions.append(finetuned_prediction)
    idx += 1
    if (idx % 100) == 0:
        print(f"Processed {idx}/{TEST} windows")

df_pred = pd.DataFrame(predictions)
df_pred.to_csv(os.path.join(source_dir, 'predictions.csv'), index=False)

df_finetuned_pred = pd.DataFrame(finetuned_predictions)
df_finetuned_pred.to_csv(os.path.join(source_dir, 'finetuned_predictions.csv'), index=False)

# df_pred = pd.read_csv(os.path.join(source_dir, 'predictions.csv'), parse_dates=['timestamp'])
predictions = df_pred[df_pred.lookahead == PDT].set_index('timestamp').sort_index()
finetuned_predictions = df_finetuned_pred[df_finetuned_pred.lookahead == PDT].set_index('timestamp').sort_index()

fig = make_subplots(rows=len(cv_tags), cols=1, subplot_titles=cv_tags)
for tag in cv_tags:
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions[f'true_{tag}'], mode='lines', name=f'{tag} true', line=dict(color="blue")), row=cv_tags.index(tag) + 1, col=1)
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions[f'pred_{tag}'], mode='lines', name=f'{tag} predicted', line=dict(color="red")), row=cv_tags.index(tag) + 1, col=1)
    fig.add_trace(go.Scatter(x=finetuned_predictions.index, y=finetuned_predictions[f'pred_{tag}'], mode='lines', name=f'{tag} predicted (finetuned)', line=dict(color="green")), row=cv_tags.index(tag) + 1, col=1)
    fig.update_xaxes(title_text="Timestamp", row=cv_tags.index(tag) + 1, col=1)
    fig.update_yaxes(title_text=tag, row=cv_tags.index(tag) + 1, col=1)
    fig.update_layout(title=f"Tag: {tag}")

    # ax = plt.subplot(len(cv_tags), 1, cv_tags.index(tag) + 1)

    # ax.scatter(predictions.index, predictions[f'true_{tag}'], label='true', color='black')
    # ax.scatter(predictions.index, predictions[f'pred_{tag}'], label='predicted', color='blue')
    # ax.scatter(finetuned_predictions.index, finetuned_predictions[f'pred_{tag}'], label='predicted (finetuned)', color='blue')
    # ax.set_title("Tag: " + tag)

fig.write_html(os.path.join(source_dir, 'graphs/test.html'))
# predictions = df_pred[df_pred.lookahead == PDT].set_index('timestamp').sort_index()

# plt.show()
