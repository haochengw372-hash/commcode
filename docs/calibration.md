# Optional offline calibration

Calibration is a supervised step, separate from runtime coding. It uses final
reference labels for explicitly selected calibration IDs; it does not require
human labels for intermediate concepts or modify the original categories.

The implementation is available from `commcode.system1.calibration`:

```python
from commcode.system1.calibration import Calibrator, fit_calibration

calibrator = fit_calibration(
    book, predictions, references,
    fit_ids=calibration_ids,
    group_by_id=source_group_by_id,
    heldout_ids=evaluation_ids,
)
frozen = calibrator.to_dict()
restored = Calibrator.from_dict(frozen)
calibrated_predictions = restored.apply_rows(new_predictions)
```

`predictions` are typed System 1 outputs. Reference units use `{"unit_id": "id", "dimensions": {"dimension_id":
{"label": "original_label", "valid": true}}}`. For numeric targets use `value`
in place of `label`; Noul fractions additionally require positive `n_raters`
(or `n_valid`). Missing or masked references are excluded. Declare
source groups such as articles or conversations so related units cannot appear in
both calibration and held-out evaluation. Preserve the fitted artifact and its
input provenance with the study. The transform's integrity hash detects changes;
it does not establish the scientific validity of the chosen references.

Identity is among the candidates and is retained when fitting is unsupported or
insufficiently justified. Binary decision-threshold fitting is disabled by default
and requires explicit `fit_thresholds=True`. Report supervised adaptation separately
from zero-shot results and use the same label budget in fair comparisons.

Noul response-fraction targets require their actual valid-rater denominators.
Numeric Score targets retain their original range. A model's confidence and the
largest category probability are distinct quantities; `confidence` retains provider confidence when supplied, while
`calibrated_pmax` and `calibrated_label_probability` identify the maximum and
selected-label posterior probabilities separately.
