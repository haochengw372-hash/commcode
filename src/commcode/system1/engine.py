"""Small generic encoder; selection and calibration belong to explicit callbacks."""

from pathlib import Path

from .models import ValidationError, finite_number
from .parsing import parse_jev_response, parse_llm_response
from .providers import Journal, atomic_json
from .requests import build_jev_request, build_llm_request, cells


class System1Encoder:
    def __init__(self, provider, llm_provider=None):
        self.provider = provider
        self.llm_provider = llm_provider

    def encode(
        self,
        packet,
        book,
        policy=None,
        output_dir="commcode_run",
        *,
        targets=None,
        prompt_family="compact",
        coder="A",
        cache_epoch="v1",
        selector=None,
        postprocess=None,
    ):
        """Encode once, optionally replace selected cells, and persist raw and derived results.

        postprocess calibrates base-provider predictions before selector sees them.
        selector receives those predictions and returns explicit target pairs. LLM
        replacement outputs retain their own labels and never pass through the base
        postprocess callback. No reference labels are accepted by this runtime API.
        """
        if policy is not None:
            if not isinstance(policy, dict):
                raise ValidationError("Policy must be explicit configuration")
            prompt_family = policy.get("prompt_family", prompt_family)
            coder = policy.get("coder", coder)
        output_dir = Path(output_dir)
        journal = Journal(output_dir / "journal")
        builder = build_jev_request if self.provider.kind == "jev" else build_llm_request
        parser = parse_jev_response if self.provider.kind == "jev" else parse_llm_response
        body = builder(packet, book, targets, prompt_family, coder, self.provider.model)
        record = self.provider.invoke(body, journal, cache_epoch)
        try:
            parser_options = {"prompt_family": prompt_family} if self.provider.kind == "jev" else {}
            predictions = parser(
                self.provider.response(record), packet, book, targets, **parser_options
            )
            stages = [record]
            if postprocess is not None:
                predictions = postprocess(predictions)
                expected_base = {
                    (u["unit_id"], dim.id) for _, _, u, dim in cells(packet, book, targets)
                }
                actual_base = [(x["unit_id"], x["dimension_id"]) for x in predictions]
                if len(set(actual_base)) != len(actual_base) or set(actual_base) != expected_base:
                    raise ValidationError("Base postprocessing changed requested coverage")
            if selector is not None:
                selected = selector(predictions)
                if selected:
                    requested = {(x["unit_id"], x["dimension_id"]) for x in predictions}
                    if any((x["unit_id"], x["dimension_id"]) not in requested for x in selected):
                        raise ValidationError("Selector requested cells outside initial coverage")
                    if self.llm_provider is None:
                        raise ValidationError("Selected targets require an LLM provider")
                    replacement_body = build_llm_request(
                        packet, book, selected, prompt_family, "A", self.llm_provider.model
                    )
                    replacement_record = self.llm_provider.invoke(
                        replacement_body, journal, cache_epoch
                    )
                    replacements = parse_llm_response(
                        self.llm_provider.response(replacement_record), packet, book, selected
                    )
                    replacements = {(x["unit_id"], x["dimension_id"]): x for x in replacements}
                    predictions = [
                        replacements.get((x["unit_id"], x["dimension_id"]), x) for x in predictions
                    ]
                    stages.append(replacement_record)
            expected = {(u["unit_id"], dim.id) for _, _, u, dim in cells(packet, book, targets)}
            pairs = [(x["unit_id"], x["dimension_id"]) for x in predictions]
            if len(set(pairs)) != len(pairs) or set(pairs) != expected:
                raise ValidationError("Postprocessing changed requested target coverage")
            definitions = {dim.id: dim for dim in book.dimensions}
            for row in predictions:
                dim = definitions[row["dimension_id"]]
                if dim.kind == "choice":
                    if row.get("label") not in dim.labels:
                        raise ValidationError("Postprocessed label outside original categories")
                else:
                    value = finite_number(row.get("value"), "postprocessed value")
                    lo, hi = (0, 1) if dim.kind == "noul" else dim.output_range
                    if not lo <= value <= hi:
                        raise ValidationError("Postprocessed value outside original range")
        except Exception as exc:
            atomic_json(
                output_dir / "result.json",
                {
                    "status": "review_required",
                    "error_type": type(exc).__name__,
                    "record": record,
                    "predictions": [],
                },
            )
            raise
        result = {
            "status": "completed",
            "predictions": predictions,
            "record": record,
            "stages": stages,
        }
        atomic_json(output_dir / "result.json", result)
        return result
