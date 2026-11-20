"""Fixed, trial-balanced linear diagnostics and explicit undefined metrics."""
import hashlib
import json
import warnings
from collections import Counter

import numpy as np
from data import TARGETS, eligible
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

REGRESSION = TARGETS[:3]


def trial_weights(trials, labels=None):
    """Each trial totals one; classification balances classes within each trial.

    Constant-per-trial contrast cannot be globally class-balanced while keeping
    every trial equal. It retains equal trial weight and reports this limitation.
    """
    weights = np.empty(len(trials), dtype=float)
    for tid in np.unique(trials):
        mask = trials == tid
        if labels is None:
            weights[mask] = 1 / mask.sum()
        else:
            classes, counts = np.unique(labels[mask], return_counts=True)
            for c, n in zip(classes, counts):
                weights[mask & (labels == c)] = 1 / (len(classes) * n)
    return weights


def weighted_median(y, weights):
    order = np.argsort(y, kind='stable')
    return float(y[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]])


def regression(y, prediction, weights):
    error = prediction - y
    mse = float(np.average(error ** 2, weights=weights))
    variance = float(np.average((y - np.average(y, weights=weights)) ** 2, weights=weights))
    return {'mae': float(np.average(np.abs(error), weights=weights)), 'rmse': float(np.sqrt(mse)),
                'r2': 1-mse/variance if len(y) >= 2 and variance > 1e-20 else None,
                'r2_undefined_reason': None if len(y) >= 2 and variance > 1e-20 else 'Fewer than two values or constant target'}


def classification(y, prediction, weights, classes):
    matrix = confusion_matrix(y, prediction, labels=classes, sample_weight=weights)
    true, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    recall = np.divide(diagonal, true, out=np.full(len(classes), np.nan), where=true > 0)
    denominator = true + predicted
    f1 = np.divide(2 * diagonal, denominator, out=np.full(len(classes), np.nan), where=denominator > 0)
    return {'balanced_accuracy': float(np.nanmean(recall)), 'macro_f1': float(np.nanmean(f1)),
        'per_class_recall': {str(c): float(r) if np.isfinite(r) else None for c, r in zip(classes, recall)},
        'confusion_matrix': matrix.tolist(), 'classes': classes,
        'prevalence': {str(c): float(n / true.sum()) for c, n in zip(classes, true)},
        'undefined_recall_classes': [c for c, n in zip(classes, true) if n == 0],
        'undefined_f1_classes': [c for c, n in zip(classes, denominator) if n == 0],
        'undefined_reason': 'Absent true classes have null recall; classes absent in truth and prediction have null F1 and are omitted from macro averages.'}


def evaluate(y, prediction, trials, target, classes):
    score = regression if target in REGRESSION else lambda y, p, w: classification(y, p, w, classes)
    primary = score(y, prediction, trial_weights(trials))
    pooled = score(y, prediction, np.ones(len(y)))
    per_trial = {str(int(tid)): score(y[trials == tid], prediction[trials == tid], np.ones(np.sum(trials == tid))) for tid in np.unique(trials)}
    return {'trial_balanced': primary, 'pooled': pooled, 'per_trial': per_trial}


def support(rows, mask, target):
    selected = [r for r, yes in zip(rows, mask) if yes]
    classes = sorted({r[target] for r in selected}) if target not in REGRESSION else []
    return {'observations': len(selected), 'trials': len({r['trial_id'] for r in selected}),
        'per_trial': dict(Counter(str(r['trial_id']) for r in selected)),
        'per_trial_classes': {str(tid): dict(Counter(str(r[target]) for r in selected if r['trial_id'] == tid))
                           for tid in sorted({r['trial_id'] for r in selected})} if classes else {},
        'classes': {str(c): {'observations': sum(r[target] == c for r in selected),
                             'trials': len({r['trial_id'] for r in selected if r[target] == c})} for c in classes},
        'by_category': dict(Counter(r['category'] for r in selected)),
        'by_contrast': dict(Counter(str(r['contrast_value']) for r in selected))}


def fit_stage(stage, view, output):
    rows = view['rows']
    trials = np.array([r['trial_id'] for r in rows])
    splits = np.array([r['split'] for r in rows])
    contrast = np.array([r['contrast_value'] for r in rows])
    magnitude = np.array([r['absolute_displacement'] for r in rows])
    metrics, supports, comparisons, predictions = {}, {}, {}, {}
    model_dir = output / 'models'
    model_dir.mkdir(exist_ok=True)
    for target in TARGETS:
        eligible_mask = np.array([eligible(r, target) for r in rows])
        train = eligible_mask & (splits == 'train')
        val = eligible_mask & (splits == 'val')
        classes = list(range(5)) if target == 'contrast' else [0, 1]
        y = np.array([r[target] for r in rows])
        supports[target] = {s: support(rows, eligible_mask & (splits == s), target) for s in ('train', 'val')}
        supports[target]['exclusions'] = {s: dict(Counter(
            'off_screen_center' if not r['on_screen'] else 'blank_or_unrepresented' if not r['nonblank'] else r['category']
            for r, yes in zip(rows, eligible_mask) if not yes and r['split'] == s)) for s in ('train', 'val')}
        anchors = [(r['trial_id'], r['anchor']) for r, yes in zip(rows, eligible_mask) if yes]
        anchor_hash = hashlib.sha256(json.dumps(anchors, separators=(',', ':')).encode()).hexdigest()
        supports[target]['shared_ABC_anchor_sha256'] = anchor_hash
        if not train.any() or not val.any() or (target not in REGRESSION and len(np.unique(y[train])) < 2):
            metrics[target] = {'unavailable': 'Insufficient training/validation support or fewer than two training classes'}
            continue
        w = trial_weights(trials[train], y[train] if target not in REGRESSION else None)
        # Mean sample weight one keeps alpha/C on a documented sample scale;
        # multiplying all trial totals equally preserves trial equality.
        w *= len(w) / w.sum()
        base_w = trial_weights(trials[train])
        if target in REGRESSION:
            mean = float(np.average(y[train], weights=base_w))
            median = weighted_median(y[train], base_w)
            baselines = {name: evaluate(y[val], np.full(val.sum(), value), trials[val], target, classes)
                         for name, value in dict(training_mean=mean, training_median=median,
                             **({'zero_displacement': 0.} if target == 'signed_displacement' else {})).items()}
        else:
            prevalence = np.array([base_w[y[train] == c].sum() for c in classes]) / base_w.sum()
            majority = int(classes[np.argmax(prevalence)])
            baselines = {'majority': evaluate(y[val], np.full(val.sum(), majority), trials[val], target, classes),
                'majority_class': majority, 'training_trial_balanced_prevalence': prevalence.tolist(),
                'uniform_chance_balanced_accuracy': 1 / len(classes),
                'chance_note': 'Expected balanced accuracy for independent uniform guesses over the declared classes; no random baseline is fitted.'}
        result = {'baselines': baselines, 'variants': {}}
        validation_predictions = {}
        # Motion magnitude bins are frozen from train only, for all A/B/C.
        magnitude_edges = np.unique(np.quantile(magnitude[train], [.25, .5, .75])).tolist()
        supports[target]['training_magnitude_edges_deg'] = magnitude_edges
        for variant in ('A', 'B', 'C'):
            print(f'Fit {stage}/{target}/{variant}: train={train.sum()} val={val.sum()}', flush=True)
            def matrix(mask, variant=variant):
                if variant == 'A':
                    return view['static'][mask].astype(np.float64)
                if variant == 'B':
                    return view['difference'][mask].astype(np.float64)
                return np.concatenate([view['static'][mask], view['difference'][mask]], axis=1).astype(np.float64)
            xtrain, xval = matrix(train), matrix(val)
            scaler = StandardScaler()
            scaler.fit(xtrain, sample_weight=w)
            xtrain = scaler.transform(xtrain, copy=False)
            xval = scaler.transform(xval, copy=False)
            model = (Ridge(alpha=1., solver='cholesky') if target in REGRESSION else
                     LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-4, random_state=0))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(xtrain, y[train], sample_weight=w)
            prediction = model.predict(xval)
            validation_predictions[variant] = prediction
            scores = evaluate(y[val], prediction, trials[val], target, classes)
            if target in REGRESSION:
                scores['improvement'] = {name: {
                    'mae_reduction': baseline['trial_balanced']['mae'] - scores['trial_balanced']['mae'],
                    'rmse_reduction': baseline['trial_balanced']['rmse'] - scores['trial_balanced']['rmse'],
                    'mae_ratio': scores['trial_balanced']['mae']/baseline['trial_balanced']['mae'] if baseline['trial_balanced']['mae'] > 0 else None}
                    for name, baseline in baselines.items()}
            scores['strata'] = {}
            for name, values in [('contrast', contrast[val]), ('magnitude_quartile_train', np.digitize(magnitude[val], magnitude_edges))]:
                scores['strata'][name] = {str(c): {'observations': int(np.sum(values == c)),
                    'trials': len(np.unique(trials[val][values == c])),
                    'scores': evaluate(y[val][values == c], prediction[values == c], trials[val][values == c], target, classes)}
                    for c in np.unique(values)}
            if target == 'visible_motion':
                controls = (splits == 'val') & np.array([r['category'] in ('identical_nonzero_command', 'true_blank', 'visible_stationary') for r in rows])
                cp = model.predict(scaler.transform(matrix(controls)))
                cr = [r for r, yes in zip(rows, controls) if yes]
                scores['unchanged_pixel_controls'] = {}
                categories = np.array([r['category'] for r in cr])
                for c in np.unique(categories):
                    mask = categories == c
                    scores['unchanged_pixel_controls'][c] = {'observations': int(mask.sum()),
                        'trials': len(np.unique(trials[controls][mask])),
                        'false_positive_rate': float(np.average(cp[mask] == 1, weights=trial_weights(trials[controls][mask]))),
                        'pooled_false_positive_rate': float(np.mean(cp[mask] == 1))}
            scores.update(anchor_sha256=anchor_hash, convergence_warnings=[str(w.message) for w in caught],
                          iterations=getattr(model, 'n_iter_', None).tolist() if getattr(model, 'n_iter_', None) is not None else None)
            np.savez_compressed(model_dir / f'{stage}-{target}-{variant}.npz',
                mean=scaler.mean_, scale=scaler.scale_, coefficients=model.coef_, intercept=model.intercept_,
                classes=getattr(model, 'classes_', np.array([])))
            result['variants'][variant] = scores
        metrics[target] = result
        predictions[target] = dict(trial=trials[val], anchor=np.array([r['anchor'] for r, yes in zip(rows, val) if yes]),
            time=np.array([r['time'] for r, yes in zip(rows, val) if yes]), target=y[val], contrast=contrast[val],
            magnitude=magnitude[val], **validation_predictions)
        np.savez_compressed(output / f'predictions-{stage}-{target}.npz', **predictions[target])
        metric_names = ['mae', 'rmse', 'r2'] if target in REGRESSION else ['balanced_accuracy', 'macro_f1']
        comparisons[target] = {'anchor_sha256': anchor_hash,
            'differences': {f'{b}_minus_{a}': {name: (
                result['variants'][b]['trial_balanced'][name] - result['variants'][a]['trial_balanced'][name]
                if result['variants'][a]['trial_balanced'][name] is not None and result['variants'][b]['trial_balanced'][name] is not None else None)
                for name in metric_names} for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]},
            'note': 'Descriptive differences on exactly matched validation anchors; no IID frame significance claims.'}
    return metrics, supports, comparisons, predictions
