"""Execute E1 against retained train/validation artifacts from the checkout root."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from data import assemble, eligible, verify_integrity
from probes import REGRESSION, fit_stage, trial_weights
from threadpoolctl import threadpool_limits

from utils.provenance import file_hash


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')


def corr(x, y, w):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x-np.average(x, weights=w), y-np.average(y, weights=w)
    denominator = np.sqrt(np.average(x*x, weights=w)*np.average(y*y, weights=w))
    return float(np.average(x*y, weights=w)/denominator) if denominator > 1e-15 else None


def confounds(views, provenance):
    result = {}
    for stage, view in views.items():
        rows = view['rows']
        entry = {'categories': {}, 'direction_overlap': {}, 'correlations': {}, 'stationary_runs': {}, 'spatial_content': {}}
        for split in ('train', 'val'):
            selected = [r for r in rows if r['split'] == split]
            entry['categories'][split] = {category: {'observations': sum(r['category'] == category for r in selected),
                'trials': len({r['trial_id'] for r in selected if r['category'] == category})} for category in sorted({r['category'] for r in selected})}
            motion = [r for r in selected if eligible(r, 'direction')]
            w = trial_weights(np.array([r['trial_id'] for r in motion]))
            entry['correlations'][split] = {f'{a}__{b}': corr([r[a] for r in motion], [r[b] for r in motion], w)
                for a, b in [('position', 'direction'), ('initial_side', 'direction'), ('position', 'initial_side'),
                             ('contrast_value', 'absolute_displacement'), ('phase', 'direction')]}
            for transform in (np.sin, np.cos):
                entry['correlations'][split][f'phase_{transform.__name__}__direction'] = corr(
                    [transform(r['phase']) for r in motion], [r['direction'] for r in motion], w)
            bins = defaultdict(lambda: {0: set(), 1: set()})
            counts = defaultdict(Counter)
            for r in motion:
                key = f"contrast={r['contrast_value']};position_bin10deg={int(np.floor(r['position']/10))}"
                bins[key][r['direction']].add(r['trial_id'])
                counts[key][r['direction']] += 1
            overlapping = {key for key, values in bins.items() if min(len(values[d]) for d in (0, 1)) >= 3}
            entry['direction_overlap'][split] = {
                'rule': 'Fixed 10-degree position bins crossed with generator contrast; at least three distinct trials per direction.',
                'strata': {key: {'left_trials': len(values[0]), 'right_trials': len(values[1]),
                    'left_pairs': counts[key][0], 'right_pairs': counts[key][1], 'sufficient': key in overlapping} for key, values in bins.items()},
                'sufficiently_overlapping_strata': len(overlapping), 'total_strata': len(bins),
                'fraction_pairs_within_overlap': sum(counts[k][0]+counts[k][1] for k in overlapping)/len(motion) if motion else None,
                'interpretation': 'Confounded: aggregate direction accuracy does not establish intrinsic motion decoding; see sparse and nonoverlapping strata and position/side correlations.'}
            longest, current, previous = 0, 0, None
            for r in selected:
                key = (r['trial_id'], r['anchor'])
                consecutive = previous is not None and key == (previous[0], previous[1]+1)
                current = current+1 if consecutive and r['category'] == 'visible_stationary' else int(r['category'] == 'visible_stationary')
                longest = max(longest, current)
                previous = key
            entry['stationary_runs'][split] = {'longest_visible_stationary_pair_run': longest,
                'note': 'Adjacent pairs overlap and are temporally dependent; trials, not pairs, are the generalization unit.'}
            entry['spatial_content'][split] = {'off_screen_center_pairs': sum(not r['on_screen'] for r in selected),
                'nonblank_pairs': sum(r['nonblank'] for r in selected),
                'nonzero_center_pairs': sum(abs(r['position']) > 1e-10 for r in selected),
                'note': 'Center outside the source ViewWindow is excluded from spatial/motion probes. The finite source quad and Gabor tails can be clipped even for an on-screen center; no retinal or optical-flow ground truth is claimed.'}
        entry['interpolation'] = ('Not applicable: normalized source observations.' if stage == 'source' else
            'Adjacent centers normally share a source frame. Both endpoint brackets and all intervening source intervals must be smooth; direction reversals and moving identical-pixel constituents are excluded from positives. Center labels are interpolated reconstruction states, not new pixel measurements.')
        result[stage] = entry
    summaries = provenance['trial_summary']
    result['trial_associations'] = {}
    for split in ('train', 'val'):
        ss = [r for r in summaries if r['split'] == split]
        motion_counts = Counter(r['trial_id'] for r in views['source']['rows'] if r['split'] == split and r['category'] == 'visible_motion')
        fractions = [motion_counts[r['trial_id']]/max(1, r['source_observations']-1) for r in ss]
        result['trial_associations'][split] = {
            'contrast_duration': corr([r['contrast'] for r in ss], [r['duration'] for r in ss], np.ones(len(ss))),
            'duration_motion_fraction': corr([r['duration'] for r in ss], fractions, np.ones(len(ss))),
            'contrast_motion_fraction': corr([r['contrast'] for r in ss], fractions, np.ones(len(ss))),
            'unique_phase_values': len({r['phase'] for r in ss}), 'trials': len(ss)}
    result['shortcuts'] = 'Synthetic phase is fixed within a trial and identifies it; phase is not an input to the CLIP probes. Held-out trials prevent direct within-trial memorization but do not remove trajectory/phase distribution shortcuts. No phase-invariant generalization claim.'
    return result


def plots(stage, metrics, predictions, output):
    folder = output / 'plots'
    folder.mkdir(exist_ok=True)
    for target, p in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for ax, variant in zip(axes, ('A', 'B', 'C')):
            if target in REGRESSION:
                # Deterministic chronological display thinning only. Every
                # eligible validation anchor contributes to numeric metrics.
                ids = np.linspace(0, len(p['target'])-1, min(2500, len(p['target']))).astype(int)
                ax.scatter(p['target'][ids], p[variant][ids], c=p['contrast'][ids], s=3, alpha=.3)
                lo, hi = np.min(p['target']), np.max(p['target'])
                ax.plot([lo, hi], [lo, hi], color='black', linewidth=.8)
                ax.set(xlabel='Target (deg)', ylabel='Prediction (deg)')
            else:
                score = metrics[target]['variants'][variant]['trial_balanced']
                matrix = np.array(score['confusion_matrix'])
                matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True)>0)
                ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
                labels = ['0', '0.0625', '0.125', '0.25', '1'] if target == 'contrast' else ['left', 'right'] if target == 'direction' else ['stationary', 'moving']
                ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel='Predicted', ylabel='True')
                ax.tick_params(axis='x', labelrotation=40)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
            ax.set_title(variant)
        fig.suptitle(f'{stage}: {target}; validation')
        fig.savefig(folder / f'{stage}-{target}.png', dpi=130)
        plt.close(fig)
        if target in REGRESSION:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
            for ax, (name, values) in zip(axes, [('contrast', p['contrast']), ('motion magnitude (deg)', np.digitize(p['magnitude'], metrics[target].get('magnitude_edges', [])))]):
                if name.startswith('motion'):
                    # Fixed physical bins for plotting, independent of val labels.
                    values = np.digitize(p['magnitude'], [.01, .1, 1., 10.])
                groups = np.unique(values)
                for variant in ('A', 'B', 'C'):
                    errors = [np.average(np.abs(p[variant][values == c]-p['target'][values == c]), weights=trial_weights(p['trial'][values == c])) for c in groups]
                    ax.plot(range(len(groups)), errors, marker='o', label=variant)
                ax.set(xticks=range(len(groups)), xticklabels=[str(c) for c in groups], xlabel=name if name == 'contrast' else 'Magnitude bin: <.01, .01–.1, .1–1, 1–10, >=10 deg', ylabel='Trial-balanced MAE (deg)')
                ax.legend()
            fig.suptitle(f'{stage}: {target} errors; validation')
            fig.savefig(folder / f'{stage}-{target}-strata.png', dpi=130)
            plt.close(fig)


def report(metrics, supports, provenance, diagnostics, output):
    lines = ['# E1 — Retained visual information probes', '',
        'This is a within-session, trial-held-out train/validation diagnostic of retained normalized CLIP features. No neural model, production feature extraction, stimulus generation, alignment or split membership was changed.', '',
        '## Data and verification', '',
        f"Dataset `{provenance['dataset_generation']}`, session `{provenance['session']}`. Frozen membership: 218 train, 31 validation, 63 test. Test payloads were accessed only inside integrity readers with explicit user authorization; no test observations enter experimental labels, fitting, diagnostics or plots. The complete stored membership is recorded in `manifest.json`.", '',
        provenance['verification']['boundary'], '',
        'ReplayArtifactReader and FeatureArtifactReader are exhausted, and load_alignment and load_dataset complete with the expected generation identities. Their test payload access is confined to integrity checks; values are discarded rather than returned to experiment views. The train/validation view additionally checks selected payload hashes, replay/feature metadata joins, source identities and saved interpolation directly.', '',
        f"All {len(provenance['verification']['interpolation'])} selected aligned trials reproduce their saved float32 feature values exactly from saved source associations. Representative lossless RGB pair checks and previews cover the retained categories (`control-*.png`).", '',
        '## Fixed protocol', '',
        'A is the current feature, B is its unnormalized first difference, and C concatenates A and B. All three use the same anchors per target/stage. Consecutive physical intervals must be 1/60 s within 1e-8 s. Differences cannot cross trial, gap, event-stage, visibility, parameter or wrap boundaries. Position and motion require nonblank visible content with an on-screen reconstructed center. Displacement regression includes visible stationary zero controls. Contrast uses all qualified regular pairs, including blank contrast-zero images.', '',
        'Aligned targets linearly interpolate reconstructed azimuth only within smooth source brackets; adjacent-center motion additionally requires smooth intervening source intervals without direction reversals. Positive aligned motion requires every moving constituent interval to change source pixels. These are not measured center-frame pixels or retinal trajectories.', '',
        'Ridge alpha=1; logistic C=1, lbfgs, tolerance=1e-4, maximum 1000 iterations; multinomial softmax for five-class contrast. StandardScaler is fitted only on train for each target/variant using its training weights. Difference vectors receive no per-vector L2 normalization. Each contributing trial has equal total training weight, normalized to mean sample weight one. Classification balances observed classes within each trial. Contrast is constant within trial, so global class balance cannot simultaneously be enforced with equal trial totals; contrast preserves equal trial totals. This limitation is explicit rather than silently reweighting trial classes.', '',
        'Primary metrics use inverse-count weights so every contributing validation trial totals one, then compute MAE/RMSE/R² or recalls/macro-F1 from that weighted distribution. Supplementary pooled and individual-trial metrics are retained. Undefined recall/F1/R² values are null with reasons. Baseline means, medians, majority classes and magnitude quantile edges are fitted only on train. There was no search, feature selection or threshold tuning.', '',
        '## Validation results', '',
        '| Stage | Target | Train / val trials | A | B | C | Baseline |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for stage, targets in metrics.items():
        for target, result in targets.items():
            if 'variants' not in result:
                lines.append(f'| {stage} | {target} | — | unavailable | — | — | — |')
                continue
            metric = 'mae' if target in REGRESSION else 'balanced_accuracy'
            values = [result['variants'][v]['trial_balanced'][metric] for v in ('A', 'B', 'C')]
            baseline = result['baselines']['training_mean' if target in REGRESSION else 'majority']['trial_balanced'][metric]
            coverage = supports[stage][target]
            lines.append(f"| {stage} | {target} ({'MAE deg' if target in REGRESSION else 'balanced accuracy'}) | {coverage['train']['trials']} / {coverage['val']['trials']} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {baseline:.4f} |")
    lines += ['', 'Full RMSE, R², baseline improvements/ratios, per-class recalls, confusion matrices, prevalence, trial counts, exclusions, strata and unchanged-pixel false positives are in `metrics.json` and `support.json`. Paired descriptive A/B/C differences are in `paired-comparisons.json`; validation predictions and fitted coefficients/scalers are saved separately.', '',
              '## What the probes recovered', '']
    for stage, targets in metrics.items():
        def score(target, variant, targets=targets):
            return targets[target]['variants'][variant]['trial_balanced']

        if not all('variants' in targets[t] for t in ('position', 'contrast', 'signed_displacement', 'absolute_displacement', 'visible_motion')):
            lines += [f'{stage}: some required comparisons are unavailable; see the explicit support reasons.', '']
            continue
        position_baseline = targets['position']['baselines']['training_mean']['trial_balanced']['mae']
        zero = targets['signed_displacement']['baselines']['zero_displacement']['trial_balanced']
        median = targets['absolute_displacement']['baselines']['training_median']['trial_balanced']
        signed = score('signed_displacement', 'B')
        absolute = score('absolute_displacement', 'B')
        lines += [f"**{stage}.** Static position MAE is {score('position', 'A')['mae']:.3f}° versus {position_baseline:.3f}° for the training-mean baseline (R² {score('position', 'A')['r2']:.3f}). Static/combined contrast balanced accuracies are {score('contrast', 'A')['balanced_accuracy']:.3f}/{score('contrast', 'C')['balanced_accuracy']:.3f} versus 0.200 for the majority-class baseline. These targets are accessible on the qualified held-out trials.", '',
            f"Difference-only signed-displacement MAE is {signed['mae']:.4f}° versus {zero['mae']:.4f}° for zero displacement; its RMSE is {signed['rmse']:.4f}° versus {zero['rmse']:.4f}° and R² is {signed['r2']:.3f}. Thus MAE alone and squared-error metrics can support different conclusions; zero/median controls must not be omitted. Difference-only absolute-displacement MAE is {absolute['mae']:.4f}° versus {median['mae']:.4f}° for the training-median baseline (R² {absolute['r2']:.3f}). Temporal features expose some displacement information, with remaining error.", '',
            f"Visible-motion balanced accuracy is {score('visible_motion', 'A')['balanced_accuracy']:.3f}/{score('visible_motion', 'B')['balanced_accuracy']:.3f}/{score('visible_motion', 'C')['balanced_accuracy']:.3f} for A/B/C. A strong static result indicates state/trajectory shortcuts are available. A signed linear difference need not implement the sign-symmetric magnitude decision required for motion presence; its failure is not evidence that temporal information is absent.", '']
        controls = targets['visible_motion']['variants']['C'].get('unchanged_pixel_controls', {})
        if 'identical_nonzero_command' in controls:
            control = controls['identical_nonzero_command']
            lines += [f"The combined probe calls {control['false_positive_rate']:.1%} of identical-pixel/nonzero-command controls moving under equal-control-trial weighting ({control['trials']} trials). These controls are excluded from positive training labels. High ordinary motion accuracy therefore does not establish general, intrinsic visible-motion decoding.", '']
    capped = [f'{stage}/{target}/{variant}' for stage, targets in metrics.items()
              for target, result in targets.items() for variant, scores in result.get('variants', {}).items()
              if scores['convergence_warnings']]
    lines += [('Iteration-cap/nonconvergence warnings: ' + ', '.join(capped) + '. These scores are diagnostic unfinished optimizer fits; do not interpret their ranking as a converged optimum. The fixed protocol was not tuned using validation.') if capped else 'All fits completed without recorded convergence warnings.', '',
        '## Interpretation boundaries', '',
        'Successful validation decoding establishes accessibility to these fixed linear probes on held-out trials from this reconstruction. A failed probe establishes only that this particular linear fit did not recover a target. It does not demonstrate information-theoretic loss. Identical source images cannot identify their different commanded states intrinsically, and are excluded from direction and positive visible-motion labels. Any predictions on such controls can exploit context or static confounds.', '',
        'Source and aligned scores use different physical anchor times and different smooth-interval support. They are not matched causal estimates of interpolation damage. Shared source brackets also make adjacent aligned differences dependent. No difference between stages is attributed solely to interpolation.', '',
        f"Exact RGB hashes shared across train/validation: {provenance['duplicate_images']['shared_hash_count']}; nonblank shared hashes: {provenance['duplicate_images']['shared_nonblank_hash_count']}. Duplicate content and stationary runs reduce effective image diversity. Trial balancing limits the influence of long trials but does not create independent frames.", '',
        'Direction accuracy is confounded wherever opposite directions lack shared position/contrast support. `diagnostics.json` reports fixed 10-degree/contrast strata, independent trial counts, overlap fractions and position/side/direction correlations for each split. Aggregate direction scores must not be interpreted as intrinsic motion decoding. Synthetic phase is constant within each trial and can serve as an appearance signature; this protocol does not demonstrate invariance to it. Contrast-duration and movement associations are also reported.', '',
        'Off-screen centers are excluded from spatial/motion targets. On-screen centers can still have clipped Gabor/quad content; retained pixels and schematic geometry do not establish a full unclipped retinal object. All labels inherit assumed geometry/gain/wheel sign, synthetic phase and event proxies. Vertical position does not vary and is not probed. Pre-normalization vectors/norms and original experimental framebuffer images were not retained and cannot be evaluated.', '',
        '## Next experiment', '',
        'Prioritize a targeted motion/observability investigation before changing the production representation. Compare opposite directions at shared positions and contrasts on independent trials, distinguish changes visible in retained pixels from commanded changes, and retain the current static feature as a spatial reference. The measured table above determines which target is accessible to each fixed probe. An E3 explicit-motion comparison is the next extraction candidate if motion accessibility remains inadequate after that qualification. E2 spatial-preservation changes require a concrete unresolved spatial target; feature cosine similarity alone does not justify them. Production extraction must not be changed automatically from these diagnostics.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/experiment/visual-information/e1'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=args.threads):
        integrity = verify_integrity(ROOT, args.output, workers=args.threads)
        views, provenance, exclusions = assemble(ROOT, args.output)
        provenance['verification']['public_readers'] = integrity
        provenance['verification']['complete_generation_readers_exhausted'] = True
        provenance['verification']['boundary'] = 'Full public-reader integrity verification, including authorized integrity-only access to test payloads. Scientific views and all experimental decisions use train/validation only.'
        provenance['protocol'] = {'variants': {'A': 'current', 'B': 'current-previous', 'C': 'concatenate(A,B)'},
            'ridge_alpha': 1., 'logistic_C': 1., 'logistic_tolerance': 1e-4, 'logistic_max_iterations': 1000,
            'feature_scaling': 'weighted train-only StandardScaler, independently per target and variant',
            'training_weights': 'equal total per trial; class-balanced within trial; mean weight one',
            'primary_evaluation': 'inverse-count per-trial weights; aggregate metrics on weighted observations',
            'seed': 0, 'threads': args.threads, 'regular_interval_seconds': 1/60, 'interval_atol_seconds': 1e-8,
            'stationary_atol_deg': 1e-10, 'test_used_for_experiment': False, 'test_integrity_access': True,
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'matplotlib')},
            'sources': {str(p.relative_to(ROOT)): file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
        snapshot = args.output / 'implementation'
        snapshot.mkdir()
        for source in Path(__file__).parent.glob('*.py'):
            (snapshot / source.name).write_bytes(source.read_bytes())
        write(args.output / 'manifest.json', provenance)
        diagnostics = confounds(views, provenance)
        write(args.output / 'diagnostics.json', diagnostics)
        metrics, supports, paired = {}, {}, {}
        for stage, view in views.items():
            with (args.output / f'anchors-{stage}.jsonl').open('w', encoding='utf-8') as stream:
                for row in view['rows']:
                    stream.write(json.dumps(plain(row), allow_nan=False) + '\n')
            m, s, c, p = fit_stage(stage, view, args.output)
            metrics[stage], supports[stage], paired[stage] = m, s, c
            write(args.output / 'metrics.json', metrics)
            write(args.output / 'support.json', {'stages': supports, 'continuity_exclusions': exclusions,
                'candidate_categories': {stage: {split: dict(Counter(r['category'] for r in view['rows'] if r['split'] == split)) for split in ('train', 'val')} for stage, view in views.items()}})
            write(args.output / 'paired-comparisons.json', paired)
            plots(stage, m, p, args.output)
        report(metrics, supports, provenance, diagnostics, args.output)
    print(f'E1 complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
