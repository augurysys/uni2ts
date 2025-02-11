import copy
import os

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator_ml_utils.eval.metrics import performance_metrics

def compare_results(source_dir: str, checkpoint_path: str, context_length: int, prediction_length: int,
                    batch_size: int = 6, patch_size: (int|str) = "auto",
                    model_type: str = "moirai", model_size: str = "small", test_size: (int|None) = None):
    graphs_dir = os.path.join(source_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(source_dir, "extracted_features.parquet"))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    if test_size is not None:
        df = df.head(test_size)
    else:
        test_size = len(df)

    factory_cfg = pd.read_csv(os.path.join(source_dir, "factory-control-loop-tag.csv.xz"))
    cv_tags = [str(c) for c in factory_cfg[factory_cfg.tag_type == "CV"].tag_id.unique()]
    mv_dv_tags = [str(c) for c in factory_cfg[factory_cfg.tag_type != "CV"].tag_id.unique()]

    try:
        simulator_predictions_df = pd.read_csv(os.path.join(source_dir, "test_forecast_eval.csv.xz"), parse_dates=['timestamp'])
        simulator_predictions_df = simulator_predictions_df[simulator_predictions_df.lookahead == prediction_length].set_index('timestamp').sort_index()
    except FileNotFoundError:
        simulator_predictions_df = None


    ds = PandasDataset(
        df,
        target=cv_tags if len(cv_tags) > 1 else cv_tags[0],
        timestamp="timestamp",
        feat_dynamic_real=mv_dv_tags,
        freq="1min",
        unchecked=True
    )

    # Split into train/test set
    train, test_template = split(
        ds, offset=-test_size
    )  # assign last TEST time steps as test set

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,  # number of time steps for each prediction
        windows=test_size-prediction_length+1,  # number of windows in rolling window evaluation
        distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
    )

    # Prepare pre-trained model by downloading model weights from hugg1ingface hub
    if model_type == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{model_size}"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=100,
            target_dim=len(cv_tags),
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        finetuned_model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            prediction_length=prediction_length,
            target_dim=len(cv_tags),
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            context_length=context_length
        )

    elif model_type == "moirai-moe":
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{model_size}"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=100,
            target_dim=len(cv_tags),
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

    predictor = model.create_predictor(batch_size=batch_size)
    finetuned_predictor = finetuned_model.create_predictor(batch_size=batch_size)
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
        for lookahead in range(prediction_length):
            prediction = {
                "timestamp": label['start'].start_time + pd.Timedelta(minutes=lookahead),
                "lookahead": lookahead + 1,
            }
            for i, tag_id in enumerate(cv_tags):
                if len(cv_tags) == 1:
                    prediction[f"true_{tag_id}"] = np.median(label['target'][lookahead])
                    prediction[f"pred_{tag_id}"] = np.median(forecast.samples[:, lookahead])
                    prediction[f"prctile_05_{tag_id}"] = np.percentile(forecast.samples[:, lookahead], 5)
                    prediction[f"prctile_95_{tag_id}"] = np.percentile(forecast.samples[:, lookahead], 95)
                    prediction[f"prctile_25_{tag_id}"] = np.percentile(forecast.samples[:, lookahead], 25)
                    prediction[f"prctile_75_{tag_id}"] = np.percentile(forecast.samples[:, lookahead], 75)
                else:
                    prediction[f'true_{tag_id}'] = label['target'][i, lookahead]
                    prediction[f'pred_{tag_id}'] = np.median(forecast.samples[:, lookahead, i])
                    prediction[f'prctile_05_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 5)
                    prediction[f'prctile_95_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 95)
                    prediction[f'prctile_25_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 25)
                    prediction[f'prctile_75_{tag_id}'] = np.percentile(forecast.samples[:, lookahead, i], 75)
            predictions.append(prediction)
            finetuned_prediction = copy.deepcopy(prediction)
            for i, tag_id in enumerate(cv_tags):
                if len(cv_tags) == 1:
                    finetuned_prediction[f'true_{tag_id}'] = label['target'][lookahead]
                    finetuned_prediction[f'pred_{tag_id}'] = np.median(finetuned_forecast.samples[:, lookahead])
                    finetuned_prediction[f'prctile_05_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead], 5)
                    finetuned_prediction[f'prctile_95_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead], 95)
                    finetuned_prediction[f'prctile_25_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead], 25)
                    finetuned_prediction[f'prctile_75_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead], 75)
                else:
                    finetuned_prediction[f'true_{tag_id}'] = label['target'][i, lookahead]
                    finetuned_prediction[f'pred_{tag_id}'] = np.median(finetuned_forecast.samples[:, lookahead, i])
                    finetuned_prediction[f'prctile_05_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 5)
                    finetuned_prediction[f'prctile_95_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 95)
                    finetuned_prediction[f'prctile_25_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 25)
                    finetuned_prediction[f'prctile_75_{tag_id}'] = np.percentile(finetuned_forecast.samples[:, lookahead, i], 75)
            finetuned_predictions.append(finetuned_prediction)
        idx += 1
        if (idx % 100) == 0:
            print(f"Processed {idx}/{test_size} windows")

    df_pred = pd.DataFrame(predictions)
    df_pred.to_csv(os.path.join(source_dir, 'predictions.csv'), index=False)

    if simulator_predictions_df is not None:
        simulator_predictions_df = simulator_predictions_df.loc[df_pred['timestamp'].min():df_pred['timestamp'].max()]

    df_finetuned_pred = pd.DataFrame(finetuned_predictions)
    df_finetuned_pred.to_csv(os.path.join(source_dir, 'finetuned_predictions.csv'), index=False)

    df_pred = pd.read_csv(os.path.join(source_dir, 'predictions.csv'), parse_dates=['timestamp'])
    df_finetuned_pred = pd.read_csv(os.path.join(source_dir, 'finetuned_predictions.csv'), parse_dates=['timestamp'])

    predictions = df_pred[df_pred.lookahead == prediction_length].set_index('timestamp').sort_index()
    finetuned_predictions = df_finetuned_pred[df_finetuned_pred.lookahead == prediction_length].set_index('timestamp').sort_index()

    fig = make_subplots(rows=len(cv_tags), cols=1, subplot_titles=cv_tags, shared_xaxes=True)
    for tag in cv_tags:
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[f'true_{tag}'], mode='lines', name=f'{tag} true', line=dict(color="blue")), row=cv_tags.index(tag) + 1, col=1)
        if simulator_predictions_df is not None:
            fig.add_trace(go.Scatter(x=simulator_predictions_df.index, y=simulator_predictions_df[f'pred_{tag}'], mode='lines', name=f'{tag} predicted (simulator)', line=dict(color="magenta")), row=cv_tags.index(tag) + 1, col=1)
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[f'pred_{tag}'], mode='lines', name=f'{tag} predictedb (zero shot)', line=dict(color="red")), row=cv_tags.index(tag) + 1, col=1)
        fig.add_trace(go.Scatter(x=finetuned_predictions.index, y=finetuned_predictions[f'pred_{tag}'], mode='lines', name=f'{tag} predicted (finetuned)', line=dict(color="green")), row=cv_tags.index(tag) + 1, col=1)
        fig.update_xaxes(title_text="Timestamp", row=cv_tags.index(tag) + 1, col=1)
        fig.update_yaxes(title_text=tag, row=cv_tags.index(tag) + 1, col=1)
        fig.update_layout(title=f"Tag: {tag}")

    fig.write_html(os.path.join(graphs_dir, 'test.html'))

    allowed_errors = {tag: factory_cfg[factory_cfg.tag_id == int(tag)].iloc[0]['accepted_prediction_error'] for tag in cv_tags}

    actual_cols = [f'true_{tag}' for tag in cv_tags]
    pred_cols = [f'pred_{tag}'for tag in cv_tags]

    simulator_metrics = dict() if simulator_predictions_df is None else performance_metrics(
        simulator_predictions_df[actual_cols].values,
        simulator_predictions_df[pred_cols].values,
        y_header=cv_tags,
        prefix='test lookahead',
        allowed_errors=allowed_errors
    )

    zero_shot_metrics = performance_metrics(
        predictions[actual_cols].values,
        predictions[pred_cols].values,
        y_header=cv_tags,
        prefix='test lookahead',
        allowed_errors=allowed_errors
    )

    finetuned_metrics = performance_metrics(
        finetuned_predictions[actual_cols].values,
        finetuned_predictions[pred_cols].values,
        y_header=cv_tags,
        prefix='test lookahead',
        allowed_errors=allowed_errors
    )

    metrics_df = pd.DataFrame({'simulator': simulator_metrics, 'zero_shot': zero_shot_metrics, 'finetuned': finetuned_metrics})
    metrics_df.to_csv(os.path.join(source_dir, 'metrics.csv'))

    print(metrics_df.to_markdown())

# source_dir = "data/SSS/test"
# checkpoint_path = "/home/dbarsky/Code/uni2ts/outputs/finetune/moirai_1.1_R_small/SSS_dynamic_feats_train/SSS_finetune/checkpoints/epoch=4-step=5000.ckpt"
# model_type = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
# model_size = "small"  # model size: choose from {'small', 'base', 'large'}
# prediction_length = 25  # prediction length: any positive integer
# context_length = 60  # context length: any positive integer
# patch_size = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
# batch_size = 6  # batch size: any positive integer
# test_length = None  # test set length: any positive integer

source_dir = "data/Barilla/test"
checkpoint_path = "/home/dbarsky/Code/uni2ts/outputs/finetune/moirai_1.1_R_small/Barilla_dynamic_feats_train/Barilla_finetune/checkpoints/epoch=28-step=29000.ckpt"
model_type = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
model_size = "small"  # model size: choose from {'small', 'base', 'large'}
prediction_length = 30  # prediction length: any positive integer
context_length = 60  # context length: any positive integer
patch_size = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
batch_size = 6  # batch size: any positive integer
test_length = None  # test set length: any positive integer


compare_results(source_dir=source_dir, checkpoint_path=checkpoint_path, context_length=context_length,
                prediction_length=prediction_length, batch_size=batch_size, patch_size=patch_size,
                model_type=model_type, model_size=model_size, test_size=test_length)
