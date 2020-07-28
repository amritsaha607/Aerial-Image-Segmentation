import torch
import wandb

from metrics.metrics import bakeWeight, gatherMetrics
from metrics.utils import conf_operations
from utils.vis import showPredictions
from metrics.pred import predict
from utils.decorators import timer


@timer
def evaluate(loader, model, params):
    '''
        Evalaute on a dataloader
        Args:
            loader : data loader
            model : loaded model
            params : All other parameters needed
    '''

    metrics = params['metrics']
    metric_batch = params['metric_batch']
    index2name = params['i2n']
    pred_fig_indices = params['pred_fig_indices']
    mode = params['mode']

    if mode!='run':
        raise ValueError("Sorry :/, only run mode is activated for now")


    n = len(loader)
    masks, y_preds = [], []
    if 'pred' in metrics:
        vis_img, vis_mask, vis_y_pred = [], [], []
    if metric_batch is not None:
        metrics_arr, weights_arr = [], []
    logg = {}

    model.eval()
    for batch_idx, (_, _, image, mask) in enumerate(loader):
        y_pred = model(image.cuda()).detach().cpu()
        image = image.detach().cpu()

        y_preds.append(y_pred)
        masks.append(mask)

        if 'pred' in metrics:
            if batch_idx in pred_fig_indices:
                vis_img.append(image)
                vis_mask.append(mask)
                vis_y_pred.append(y_pred)

        if (metric_batch is not None) and (1+batch_idx)%metric_batch==0 or (1+batch_idx)==n:
            logg_metrics, weights = gatherMetrics(
                params=(masks, y_preds),
                metrics=metrics,
                mode='eval',
                i2n=index2name,
                get_weights=True,
            )
            metrics_arr.append(logg_metrics)
            weights_arr.append(weights)

            y_preds, masks = [], []

        n_arr = (50*(batch_idx+1))//n
        progress = 'Evaluation : [{}>{}] ({}/{})'.format(
            '='*n_arr, '-'*(50-n_arr), (batch_idx+1), n)
        print(progress, end='\r')

    print("\n")

    # Metrics
    if metric_batch is not None:
        # Calculate metrics from fly array
        logg_metrics = bakeWeight(metrics_arr, weights_arr)
        logg_metrics['eval_conf'] = conf_operations(
            logg_metrics['eval_conf'], 
            ret_type='heatmap', debug=False, i2n=index2name,
        )
        logg_metrics['eval_prob_conf'] = conf_operations(
            logg_metrics['eval_prob_conf'], 
            ret_type='heatmap', debug=False, i2n=index2name,
        )

    else:
        masks = torch.cat(masks, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        logg_metrics = gatherMetrics(
            params=(masks, y_preds),
            metrics=metrics,
            mode='eval',
            i2n=index2name,
        )

    logg.update(logg_metrics)

    # Visualizations
    if 'pred' in metrics:
        vis_img = torch.cat(vis_img, dim=0)
        vis_mask = torch.cat(vis_mask, dim=0)
        vis_y_pred = torch.cat(vis_y_pred, dim=0)
        vis_mask_pred = predict(None, None, use_cache=True, params=(vis_y_pred, False))
        pred_fig = showPredictions(
            vis_img, vis_mask, vis_mask_pred, 
            use_path=False, ret='fig', debug=False, size='auto',
            getMatch=True,
        )
        logg.update({'eval_prediction': wandb.Image(pred_fig)})

    return logg
