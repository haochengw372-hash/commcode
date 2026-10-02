"""Supervised, original-output calibration and reference-free budgeted routing.

Calibration consumes labeled calibration units only. Application and routing never
accept reference labels. Numeric rater fractions estimate a response proportion,
not a model's epistemic confidence. All fitting uses the Python standard library.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

VERSION = 'commcode-calibration-1'
EPS = 1e-9


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name} must be numeric, not bool')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _dimensions(book: Any) -> list[dict]:
    if hasattr(book, 'to_dict'):
        book = book.to_dict()
    source = book['dimensions'] if isinstance(book, dict) else book
    dims = [x.to_dict() if hasattr(x, 'to_dict') else deepcopy(x) for x in source]
    if not dims or len({d['id'] for d in dims}) != len(dims):
        raise ValueError('Dimensions must be nonempty and unique')
    for dim in dims:
        if dim['kind'] == 'choice':
            labels = dim.get('labels', [])
            if (len(labels) < 2 or len(set(labels)) != len(labels)
                    or any(not isinstance(x, str) for x in labels)):
                raise ValueError('Original distinct class labels required')
        elif dim['kind'] == 'score':
            lo, hi = [_number(x, 'output range') for x in dim['output_range']]
            if hi <= lo:
                raise ValueError('Increasing original range required')
        elif dim['kind'] != 'noul':
            raise ValueError('Unknown primitive')
    return dims


def _softmax(values: list[float]) -> list[float]:
    high = max(values)
    weights = [math.exp(x - high) for x in values]
    total = sum(weights)
    return [x / total for x in weights]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    e = math.exp(value)
    return e / (1 + e)


def _logit(value: float) -> float:
    value = min(1 - EPS, max(EPS, value))
    return math.log(value / (1 - value))


def _prediction(answer: dict, dim: dict) -> list[float] | float:
    if dim['kind'] == 'choice':
        ps = answer.get('probabilities')
        if not isinstance(ps, dict) or set(ps) != set(dim['labels']):
            raise ValueError('Probabilities must cover exactly the original class labels')
        values = [_number(ps[x], 'probability') for x in dim['labels']]
        if any(x < 0 or x > 1 for x in values) or sum(values) <= 0:
            raise ValueError('Invalid probability distribution')
        return [x / sum(values) for x in values]
    value = _number(answer['value'], 'value')
    lo, hi = (0., 1.) if dim['kind'] == 'noul' else dim['output_range']
    if not lo <= value <= hi:
        raise ValueError('Value outside original range')
    return (value - lo) / (hi - lo)


def _identity(dim: dict, reason: str = '') -> dict:
    return {'method': 'identity', 'reason': reason, 'slope': 1., 'intercept': 0.,
            'bias': [0.] * len(dim.get('labels', [])), 'threshold': None}


def _transform(x: list[float] | float, model: dict, kind: str) -> list[float] | float:
    if model['method'] == 'identity':
        return x
    if kind == 'choice':
        return _softmax([model['slope'] * math.log(max(EPS, p)) + b
                         for p, b in zip(x, model['bias'], strict=True)])
    if kind == 'noul':
        return _sigmoid(model['slope'] * _logit(x) + model['intercept'])
    return min(1., max(0., model['slope'] * x + model['intercept']))


def _loss(rows: list[dict], model: dict, kind: str) -> float:
    total = sum(r['weight'] for r in rows)
    if not total:
        return math.inf
    loss = 0.
    for row in rows:
        p = _transform(row['x'], model, kind)
        if kind == 'choice':
            value = -math.log(max(EPS, p[row['y']]))
        elif kind == 'noul':
            value = -(row['y'] * math.log(max(EPS, p))
                      + (1 - row['y']) * math.log(max(EPS, 1 - p)))
        else:
            value = (p - row['y']) ** 2
        loss += row['weight'] * value
    return loss / total


def _fit(rows: list[dict], dim: dict, penalty: float) -> dict:
    kind = dim['kind']
    if kind == 'choice':
        if set(r['y'] for r in rows) != set(range(len(dim['labels']))):
            return _identity(dim, 'missing original class support')
        bias = [0.] * len(dim['labels'])
    else:
        bias = []
    slope, intercept = 1., 0.
    total = sum(r['weight'] for r in rows)
    if not rows or total <= 0:
        return _identity(dim, 'no valid calibration references')
    # Positive slope (inverse temperature) preserves orientation; regularized biases
    # change class prevalence without inventing classes. Bounds prevent extreme fits.
    for step in range(350):
        gs, gi = penalty * (slope - 1), penalty * intercept
        gb = [penalty * b for b in bias]
        for row in rows:
            w = row['weight'] / total
            if kind == 'choice':
                logs = [math.log(max(EPS, p)) for p in row['x']]
                ps = _softmax([slope * z + b for z, b in zip(logs, bias, strict=True)])
                residual = [p - float(j == row['y']) for j, p in enumerate(ps)]
                gs += w * sum(e * z for e, z in zip(residual, logs, strict=True))
                gb = [g + w * e for g, e in zip(gb, residual, strict=True)]
            else:
                z = _logit(row['x']) if kind == 'noul' else row['x']
                pred = (_sigmoid(slope * z + intercept) if kind == 'noul'
                        else slope * z + intercept)
                err = pred - row['y']
                factor = 1 if kind == 'noul' else 2
                gs += factor * w * err * z
                gi += factor * w * err
        rate = .08 / math.sqrt(1 + step / 50)
        slope = min(4., max(.25, slope - rate * max(-10., min(10., gs))))
        intercept = min(3., max(-3., intercept - rate * max(-10., min(10., gi))))
        bias = [min(3., max(-3., b - rate * g)) for b, g in zip(bias, gb, strict=True)]
        if bias:
            center = sum(bias) / len(bias)
            bias = [b - center for b in bias]
    return {'method': {'choice': 'temperature_bias', 'noul': 'fraction_logistic',
                       'score': 'bounded_affine'}[kind],
            'slope': slope, 'intercept': intercept, 'bias': bias,
            'penalty': penalty, 'threshold': None}


def _macro_f1(truth: list[int], predicted: list[int], k: int) -> float:
    result = []
    for label in range(k):
        tp = sum(y == label and p == label for y, p in zip(truth, predicted, strict=True))
        fp = sum(y != label and p == label for y, p in zip(truth, predicted, strict=True))
        fn = sum(y == label and p != label for y, p in zip(truth, predicted, strict=True))
        result.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return sum(result) / k


def _confidence(answer: dict, transformed: Any, dim: dict) -> float | None:
    if dim['kind'] == 'choice':
        return max(transformed)
    confidence = answer.get('confidence')
    if confidence is None:
        return None
    confidence = _number(confidence, 'confidence')
    if not 0 <= confidence <= 1:
        raise ValueError('Confidence outside 0 to 1')
    return confidence


def _risk_model(rows: list[dict], oof: list[Any], dim: dict, model: dict) -> dict:
    bins = [{'n': 0, 'sum_error': 0.} for _ in range(5)]
    errors = []
    for row, p in zip(rows, oof, strict=True):
        if dim['kind'] == 'choice':
            index = max(range(len(p)), key=lambda j: p[j])
            if model.get('threshold') is not None:
                index = int(p[1] >= model['threshold'])
            error = float(index != row['y'])
        else:
            error = abs(p - row['y'])
        errors.append(error)
        confidence = _confidence(row['answer'], p, dim)
        if confidence is not None:
            bucket = min(4, int(confidence * 5))
            bins[bucket]['n'] += 1
            bins[bucket]['sum_error'] += error
    mean = sum(errors) / len(errors) if errors else None
    return {'target': ('misclassification_probability' if dim['kind'] == 'choice'
                       else 'expected_absolute_error_normalized'),
            'global_mean': mean, 'prior_strength': 5., 'bins': bins,
            'source': 'group_crossfit' if rows else 'unfitted'}


class Calibrator:
    """Frozen supervised transform. The inference API has no reference parameter."""

    def __init__(self, dimensions: list[dict], models: dict, risks: dict, provenance: dict):
        self.dimensions = deepcopy(dimensions)
        self.models = deepcopy(models)
        self.risks = deepcopy(risks)
        self.provenance = deepcopy(provenance)
        self._dims = {d['id']: d for d in self.dimensions}

    def to_dict(self) -> dict:
        result = {'version': VERSION, 'dimensions': deepcopy(self.dimensions),
                  'models': deepcopy(self.models), 'risks': deepcopy(self.risks),
                  'provenance': deepcopy(self.provenance)}
        result['sha256'] = _hash(result)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> Calibrator:
        record = deepcopy(value)
        digest = record.pop('sha256', None)
        if record.get('version') != VERSION or digest != _hash(record):
            raise ValueError('Calibration version or integrity mismatch')
        dims = _dimensions(record['dimensions'])
        if set(record['models']) != {d['id'] for d in dims}:
            raise ValueError('Calibration models do not match original dimensions')
        for dim in dims:
            model = record['models'][dim['id']]
            slope = _number(model['slope'], 'slope')
            if not .25 <= slope <= 4:
                raise ValueError('Invalid calibration slope')
            _number(model['intercept'], 'intercept')
            allowed = {'identity', {'choice': 'temperature_bias',
                                   'noul': 'fraction_logistic',
                                   'score': 'bounded_affine'}[dim['kind']]}
            if model['method'] not in allowed:
                raise ValueError('Invalid transform for original primitive')
            for bias in model['bias']:
                _number(bias, 'class intercept')
            threshold = model.get('threshold')
            if threshold is not None:
                threshold = _number(threshold, 'decision threshold')
                if dim['kind'] != 'choice' or len(dim['labels']) != 2 or not 0 < threshold < 1:
                    raise ValueError('Invalid binary decision threshold')
            if len(model['bias']) != len(dim.get('labels', [])):
                raise ValueError('Class order/length mismatch')
        return cls(dims, record['models'], record['risks'], record['provenance'])

    def apply(self, answer: dict, dimension_id: str | None = None) -> dict:
        did = dimension_id or answer['dimension_id']
        dim, model = self._dims[did], self.models[did]
        result = deepcopy(answer)
        x = _prediction(answer, dim)
        p = _transform(x, model, dim['kind'])
        if dim['kind'] == 'choice':
            result['uncalibrated_label'] = answer.get('label')
            result['uncalibrated_probabilities'] = deepcopy(answer['probabilities'])
            index = max(range(len(p)), key=lambda j: p[j])
            if model.get('threshold') is not None:
                index = int(p[1] >= model['threshold'])
            result['label'] = dim['labels'][index]
            result['probabilities'] = dict(zip(dim['labels'], p, strict=True))
            result['calibrated_pmax'] = max(p)
            result['calibrated_label_probability'] = p[index]
        else:
            lo, hi = (0., 1.) if dim['kind'] == 'noul' else dim['output_range']
            result['uncalibrated_value'] = answer['value']
            result['value'] = lo + p * (hi - lo)
        confidence = _confidence(answer, p, dim)
        risk = self.risks[did]
        mean = risk['global_mean']
        if mean is None:
            estimate = 1 - max(p) if dim['kind'] == 'choice' else 1.
            source = 'posterior_proxy' if dim['kind'] == 'choice' else 'unfitted_upper_bound'
        elif confidence is None:
            estimate, source = mean, 'crossfit_global_mean'
        else:
            bucket = risk['bins'][min(4, int(confidence * 5))]
            prior = risk['prior_strength']
            estimate = (bucket['sum_error'] + prior * mean) / (bucket['n'] + prior)
            source = 'crossfit_confidence_bin'
        result['calibration'] = {'method': model['method'],
                                 'decision_threshold': model.get('threshold'),
                                 'supervised': bool(self.provenance['fit_unit_ids'])}
        result['risk'] = estimate
        result['risk_source'] = source
        result['risk_target'] = risk['target']
        return result

    def apply_rows(self, predictions: list[dict]) -> list[dict]:
        return [self.apply(row) for row in predictions]


def fit_calibration(book: Any, predictions: list[dict], references: list[dict], *,
                    fit_ids: list[str] | set[str], group_by_id: dict | None = None,
                    heldout_ids: list[str] | set[str] = (), fit_thresholds: bool = False,
                    max_folds: int = 3) -> Calibrator:
    """Fit using explicit calibration IDs and group-separated CV (identity included).

    Noul targets need value in [0,1] and positive n_raters (or n_valid).
    Score targets retain their original range. References marked valid=False are
    excluded, never converted to a negative class. Any unseen original choice class
    forces identity, retaining all declared labels. Thresholds optimize binary
    macro-F1 only on crossfit predictions, separately from proper-loss fitting.
    """
    dims = _dimensions(book)
    selected = set(fit_ids)
    if selected & set(heldout_ids):
        raise ValueError('Calibration and heldout IDs overlap')
    if any(not isinstance(x, str) for x in selected):
        raise ValueError('Unit IDs must be strings')
    groups = group_by_id or {}
    if group_by_id is not None and not selected <= set(groups):
        raise ValueError('Every fit unit requires its source group')
    if set(groups.get(x, x) for x in selected) & set(groups.get(x, x) for x in heldout_ids):
        raise ValueError('Calibration and heldout source groups overlap')
    refs = {}
    for row in references:
        if row['unit_id'] in selected:
            if row['unit_id'] in refs:
                raise ValueError('Duplicate reference unit')
            refs[row['unit_id']] = row
    rows_by_dim = defaultdict(list)
    seen = set()
    for answer in predictions:
        uid, did = answer['unit_id'], answer['dimension_id']
        if uid not in selected:
            continue
        if (uid, did) in seen:
            raise ValueError('Duplicate calibration prediction')
        seen.add((uid, did))
        dim = next((d for d in dims if d['id'] == did), None)
        if dim is None:
            raise ValueError('Unknown original dimension')
        ref = refs.get(uid, {}).get('dimensions', {}).get(did)
        if ref is None or not ref.get('valid', True):
            continue
        x = _prediction(answer, dim)
        if dim['kind'] == 'choice':
            if ref.get('label') not in dim['labels']:
                raise ValueError('Reference label outside original classes')
            y, weight = dim['labels'].index(ref['label']), 1.
        else:
            lo, hi = (0., 1.) if dim['kind'] == 'noul' else dim['output_range']
            value = _number(ref['value'], 'reference value')
            if not lo <= value <= hi:
                raise ValueError('Reference outside original range')
            y = (value - lo) / (hi - lo)
            weight = (_number(ref.get('n_raters', ref.get('n_valid')),
                              'valid rater denominator') if dim['kind'] == 'noul' else 1.)
            if weight <= 0:
                raise ValueError('Positive valid rater denominator required')
        rows_by_dim[did].append({'unit_id': uid, 'group': str(groups.get(uid, uid)),
                                 'x': x, 'y': y, 'weight': weight, 'answer': deepcopy(answer)})
    models, risks, summaries = {}, {}, {}
    for dim in dims:
        did, kind = dim['id'], dim['kind']
        rows = sorted(rows_by_dim[did], key=lambda r: r['unit_id'])
        distinct = sorted({r['group'] for r in rows}, key=_hash)
        folds = min(max_folds, len(distinct))
        identity = _identity(dim)
        original_support = Counter(r['y'] for r in rows) if kind == 'choice' else {}
        enough = (len(rows) >= 6 and folds >= 2 and
                  (kind != 'choice' or set(original_support) == set(range(len(dim['labels'])))))
        best, oof = identity, [r['x'] for r in rows]
        scores = {'identity': _loss(rows, identity, kind)}
        fold_by_group = {g: j % folds for j, g in enumerate(distinct)} if folds else {}
        if enough:
            best_loss = scores['identity']
            for penalty in (.01, .1, 1.):
                candidate_oof = [None] * len(rows)
                possible = True
                for fold in range(folds):
                    train = [r for r in rows if fold_by_group[r['group']] != fold]
                    model = _fit(train, dim, penalty)
                    if model['method'] == 'identity':
                        possible = False
                        break
                    for j, row in enumerate(rows):
                        if fold_by_group[row['group']] == fold:
                            candidate_oof[j] = _transform(row['x'], model, kind)
                if not possible:
                    continue
                score = 0.
                weight = sum(r['weight'] for r in rows)
                for row, p in zip(rows, candidate_oof, strict=True):
                    if kind == 'choice':
                        error = -math.log(max(EPS, p[row['y']]))
                    elif kind == 'noul':
                        error = -(row['y'] * math.log(max(EPS, p))
                                  + (1-row['y']) * math.log(max(EPS, 1-p)))
                    else:
                        error = (p-row['y']) ** 2
                    score += row['weight'] * error / weight
                scores[str(penalty)] = score
                if score < best_loss - 1e-8:
                    best_loss, best = score, _fit(rows, dim, penalty)
                    oof = candidate_oof
        if not enough:
            best['reason'] = ('missing original class support' if kind == 'choice'
                              and len(original_support) < len(dim['labels'])
                              else 'insufficient calibration units/groups')
        if fit_thresholds and enough and kind == 'choice' and len(dim['labels']) == 2:
            truth = [r['y'] for r in rows]
            candidates = [0.5] + [j / 20 for j in range(1, 20) if j != 10]
            threshold = max(candidates, key=lambda t: _macro_f1(
                truth, [int(p[1] >= t) for p in oof], 2))
            if threshold != .5:
                best['threshold'] = threshold
        models[did] = best
        risks[did] = _risk_model(rows, oof, dim, best)
        summaries[did] = {'n_units': len(rows), 'n_groups': len(distinct),
                          'n_rater_responses': sum(r['weight'] for r in rows),
                          'class_support': {dim['labels'][i]: n
                                            for i, n in original_support.items()}
                          if kind == 'choice' else None, 'cv_loss': scores,
                          'loss': {'choice': 'multiclass_log_loss',
                                   'noul': 'weighted_binomial_cross_entropy',
                                   'score': 'normalized_squared_error'}[kind]}
    provenance = {'supervised_adaptation': True, 'fit_unit_ids': sorted(selected),
                  'fit_group_ids': sorted({str(groups.get(x, x)) for x in selected}),
                  'reference_sha256': _hash([refs[x] for x in sorted(refs)]),
                  'calibration_inputs_sha256': _hash(dict(rows_by_dim)),
                  'heldout_ids_not_read': sorted(set(heldout_ids)),
                  'selection': 'group_cv_proper_loss_with_identity',
                  'decision_threshold_selection': 'binary_crossfit_macro_f1'
                  if fit_thresholds else 'disabled', 'summaries': summaries}
    return Calibrator(dims, models, risks, provenance)


def select_targets(predictions: list[dict], *, packet_by_unit: dict[str, str],
                   budget_fraction: float = .25, cell_fraction: float = 1.,
                   calibrator: Calibrator | None = None) -> list[dict]:
    """Rank whole packets by maximum estimated risk and obey a strict floor cap.

    `packet_by_unit` must describe the *actual* request packets; a document with
    many batches is not one packet. Returns cells in selected packets, at most
    floor(cell_fraction * cells_in_packet) per packet. No minimum-one exception;
    a zero budget dispatches nothing. This caps request counts, not billed dollars.
    """
    budget_fraction = _number(budget_fraction, 'budget_fraction')
    cell_fraction = _number(cell_fraction, 'cell_fraction')
    if not 0 <= budget_fraction <= 1 or not 0 <= cell_fraction <= 1:
        raise ValueError('Budget fractions must lie in 0 to 1')
    if not predictions:
        return []
    packets = defaultdict(list)
    seen = set()
    for original in predictions:
        row = calibrator.apply(original) if calibrator else original
        uid, did = row['unit_id'], row['dimension_id']
        if (uid, did) in seen:
            raise ValueError('Duplicate target cell')
        seen.add((uid, did))
        if uid not in packet_by_unit:
            raise ValueError('Actual packet mapping missing')
        risk = row.get('risk')
        if risk is None:
            ps = row.get('probabilities')
            if ps and 'value' not in row:
                values = [_number(p, 'probability') for p in ps.values()]
                if any(p < 0 or p > 1 for p in values) or sum(values) <= 0:
                    raise ValueError('Invalid probability distribution')
                risk = 1 - max(values) / sum(values)
            else:
                risk = 1.
        risk = _number(risk, 'risk')
        if not 0 <= risk <= 1:
            raise ValueError('Normalized risk must lie in 0 to 1')
        pid = str(packet_by_unit[uid])
        packets[pid].append({'unit_id': uid, 'dimension_id': did, 'risk': risk,
                             'packet_id': pid})
    cap = math.floor(budget_fraction * len(packets))
    ordered = sorted(packets, key=lambda p: (-max(x['risk'] for x in packets[p]), p))
    output = []
    for pid in ordered[:cap]:
        cells = sorted(packets[pid], key=lambda x: (-x['risk'], x['unit_id'], x['dimension_id']))
        output.extend(cells[:math.floor(cell_fraction * len(cells))])
    return output
