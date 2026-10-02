import copy
import math

import pytest

from commcode.system1.calibration import Calibrator, fit_calibration, select_targets


def book(kind='choice', labels=None):
    dim = {'id': 'd', 'question': 'Original question', 'kind': kind,
           'estimand': 'category' if kind == 'choice' else 'rater_fraction',
           'labels': labels or ['No', 'Yes']}
    if kind == 'score':
        dim.update(output_range=[0, 100], estimand='continuous_rating', labels=[])
    return {'dimensions': [dim]}


def data(n=18, kind='choice'):
    predictions, refs, groups = [], [], {}
    for j in range(n):
        uid = f'u{j:02}'
        groups[uid] = f'g{j // 2}'
        if kind == 'choice':
            label = 'Yes' if j % 2 else 'No'
            predictions.append({'unit_id': uid, 'dimension_id': 'd', 'label': label,
                                'probabilities': {'Yes': .7 if j % 2 else .3,
                                                  'No': .3 if j % 2 else .7}})
            reference = {'label': label}
        elif kind == 'noul':
            p = .2 + .6 * j / (n - 1)
            predictions.append({'unit_id': uid, 'dimension_id': 'd', 'value': p})
            reference = {'value': p * .7 + .1, 'n_raters': 10}
        else:
            p = 10 + 70 * j / (n - 1)
            predictions.append({'unit_id': uid, 'dimension_id': 'd', 'value': p})
            reference = {'value': p + 10}
        refs.append({'unit_id': uid, 'dimensions': {'d': reference}})
    return predictions, refs, groups


def fit(kind='choice', **kwargs):
    preds, refs, groups = data(kind=kind)
    return fit_calibration(book(kind), preds, refs, fit_ids=list(groups),
                           group_by_id=groups, **kwargs), preds, refs


def test_roundtrip_original_class_order_and_immutable_input():
    model, preds, _ = fit()
    before = copy.deepcopy(preds[0])
    output = model.apply(preds[0])
    assert preds[0] == before
    assert list(output['probabilities']) == ['No', 'Yes']
    assert output['label'] == 'No'
    assert output['uncalibrated_probabilities'] == before['probabilities']
    restored = Calibrator.from_dict(model.to_dict())
    assert restored.apply_rows(preds) == model.apply_rows(preds)
    broken = model.to_dict()
    broken['dimensions'][0]['labels'].reverse()
    with pytest.raises(ValueError, match='integrity'):
        Calibrator.from_dict(broken)


@pytest.mark.parametrize('bad', [math.nan, math.inf, True, -.1, 1.1])
def test_invalid_probabilities_rejected(bad):
    model, preds, _ = fit()
    preds[0]['probabilities']['Yes'] = bad
    with pytest.raises(ValueError):
        model.apply(preds[0])


def test_missing_class_support_keeps_identity_and_original_classes():
    preds, refs, groups = data()
    for ref in refs:
        ref['dimensions']['d']['label'] = 'No'
    model = fit_calibration(book(), preds, refs, fit_ids=list(groups), group_by_id=groups,
                            fit_thresholds=True)
    assert model.models['d']['method'] == 'identity'
    assert model.models['d']['threshold'] is None
    assert model.models['d']['reason'] == 'missing original class support'
    assert set(model.apply(preds[0])['probabilities']) == {'No', 'Yes'}


def test_empty_fit_identity_and_zero_fit_provenance():
    model = fit_calibration(book(), [], [], fit_ids=[])
    output = model.apply({'dimension_id': 'd', 'label': 'No',
                          'probabilities': {'No': .6, 'Yes': .4}})
    assert output['label'] == 'No'
    assert output['risk'] == pytest.approx(.4)
    assert output['calibration']['supervised'] is False


def test_explicit_fit_only_and_heldout_group_exclusion():
    preds, refs, groups = data()
    chosen = list(groups)[:12]
    model = fit_calibration(book(), preds, refs, fit_ids=chosen,
                            group_by_id=groups, heldout_ids=list(groups)[12:])
    changed = copy.deepcopy(refs)
    for row in changed[12:]:
        row['dimensions']['d']['label'] = 'invalid future label'
    other = fit_calibration(book(), preds, changed, fit_ids=chosen,
                            group_by_id=groups, heldout_ids=list(groups)[12:])
    assert model.to_dict() == other.to_dict()
    assert model.provenance['summaries']['d']['n_units'] == 12
    with pytest.raises(ValueError, match='IDs overlap'):
        fit_calibration(book(), preds, refs, fit_ids=chosen, heldout_ids=[chosen[0]])
    with pytest.raises(ValueError, match='source groups overlap'):
        fit_calibration(book(), preds, refs, fit_ids=['u00'],
                        group_by_id=groups, heldout_ids=['u01'])


def test_missing_reference_and_invalid_masks_are_not_negative_labels():
    preds, refs, groups = data()
    for ref in refs[:10]:
        ref['dimensions']['d'] = {'valid': False, 'label': 'impossible'}
    model = fit_calibration(book(), preds, refs, fit_ids=list(groups), group_by_id=groups)
    assert model.provenance['summaries']['d']['n_units'] == 8


def test_fraction_loss_and_confidence_have_distinct_meanings():
    model, preds, _ = fit('noul')
    center = {'unit_id': 'new', 'dimension_id': 'd', 'value': .5}
    output = model.apply(center)
    assert model.provenance['summaries']['d']['loss'] == 'weighted_binomial_cross_entropy'
    assert model.provenance['summaries']['d']['n_rater_responses'] == 180
    assert 'confidence' not in output
    assert output['risk_source'] == 'crossfit_global_mean'
    assert output['risk'] < .2  # .5 response fraction is not .5 error/uncertainty.
    assert output['risk_target'] == 'expected_absolute_error_normalized'
    assert 0 <= output['value'] <= 1
    assert model.apply(preds[0])['uncalibrated_value'] == preds[0]['value']


def test_rater_fraction_requires_positive_denominator():
    preds, refs, groups = data(kind='noul')
    refs[0]['dimensions']['d']['n_raters'] = 0
    with pytest.raises(ValueError, match='Positive'):
        fit_calibration(book('noul'), preds, refs, fit_ids=list(groups))


def test_score_keeps_original_range_and_estimates_numeric_error():
    model, preds, _ = fit('score')
    outputs = model.apply_rows(preds)
    assert all(0 <= x['value'] <= 100 for x in outputs)
    assert model.provenance['summaries']['d']['loss'] == 'normalized_squared_error'
    assert outputs[0]['value'] > preds[0]['value']
    assert model.models['d']['method'] == 'bounded_affine'
    assert outputs[0]['risk_target'] == 'expected_absolute_error_normalized'


def test_cv_includes_identity_and_threshold_is_separate():
    model, _, _ = fit(fit_thresholds=True)
    record = model.to_dict()
    assert 'identity' in record['provenance']['summaries']['d']['cv_loss']
    assert record['provenance']['decision_threshold_selection'] == 'binary_crossfit_macro_f1'
    assert record['models']['d']['slope'] > 0


def test_packet_budget_is_strict_and_ties_deterministic():
    cells = [{'unit_id': f'u{j}', 'dimension_id': 'd', 'risk': .8}
             for j in range(10)]
    mapping = {f'u{j}': f'p{j // 2}' for j in range(10)}
    assert select_targets(cells, packet_by_unit=mapping, budget_fraction=0) == []
    assert select_targets(cells, packet_by_unit=mapping, budget_fraction=.1) == []
    result = select_targets(cells, packet_by_unit=mapping, budget_fraction=.4, cell_fraction=.5)
    assert {x['packet_id'] for x in result} == {'p0', 'p1'}
    assert len(result) == 2
    assert result == select_targets(list(reversed(cells)), packet_by_unit=mapping,
                                   budget_fraction=.4, cell_fraction=.5)
    with pytest.raises(ValueError, match='mapping'):
        select_targets(cells, packet_by_unit={})


def test_numeric_unfitted_route_never_uses_distance_from_half():
    cells = [{'unit_id': 'a', 'dimension_id': 'd', 'value': .01},
             {'unit_id': 'b', 'dimension_id': 'd', 'value': .5}]
    result = select_targets(cells, packet_by_unit={'a': 'a', 'b': 'b'}, budget_fraction=.5)
    assert result[0]['unit_id'] == 'a'  # Equal unfitted risk; ID tie, not fraction distance.


def test_provider_confidence_is_preserved_and_pmax_is_distinct():
    model, preds, _ = fit()
    preds[0]['confidence'] = .23
    output = model.apply(preds[0])
    assert output['confidence'] == .23
    assert output['calibrated_pmax'] == max(output['probabilities'].values())
    assert output['calibrated_label_probability'] == output['probabilities'][output['label']]
    without_provider = model.apply({k: v for k, v in preds[0].items() if k != 'confidence'})
    assert 'confidence' not in without_provider
