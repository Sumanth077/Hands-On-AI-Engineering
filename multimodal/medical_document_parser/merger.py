from __future__ import annotations

from schemas import ClinicalProfile, ImagingResult, LabFinding, PatientInfo


def _merge_patient(current: PatientInfo, incoming: PatientInfo) -> PatientInfo:
    """Merge two PatientInfo objects, preferring the first non-empty value for each field."""
    return PatientInfo(
        name=current.name or incoming.name,
        age=current.age or incoming.age,
        date=current.date or incoming.date,
    )


def _dedupe_lab_findings(findings: list[LabFinding]) -> list[LabFinding]:
    """Deduplicate lab findings by test name, keeping the most severe status."""
    seen: dict[str, LabFinding] = {}
    for finding in findings:
        key = finding.test.strip().lower()
        if not key:
            continue
        existing = seen.get(key)
        if existing is None or (existing.status == "normal" and finding.status != "normal"):
            seen[key] = finding
    return list(seen.values())


def _dedupe_strings(items: list[str]) -> list[str]:
    """Remove duplicate strings from a list using case-insensitive comparison while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def merge_profiles(profiles: list[ClinicalProfile]) -> ClinicalProfile:
    """Combine per-page ClinicalProfiles into a single deduplicated profile."""
    merged = ClinicalProfile()

    for profile in profiles:
        merged.patient = _merge_patient(merged.patient, profile.patient)
        merged.lab_findings.extend(profile.lab_findings)
        merged.imaging_results.extend(profile.imaging_results)
        merged.clinical_signals.extend(profile.clinical_signals)
        merged.flagged_items.extend(profile.flagged_items)

    merged.lab_findings = _dedupe_lab_findings(merged.lab_findings)
    merged.clinical_signals = _dedupe_strings(merged.clinical_signals)
    merged.flagged_items = _dedupe_strings(merged.flagged_items)

    for finding in merged.lab_findings:
        if finding.status in {"abnormal", "critical"}:
            label = f"{finding.test}: {finding.value}"
            if finding.normal_range:
                label += f" (ref: {finding.normal_range})"
            label += f" [{finding.status}]"
            if label not in merged.flagged_items:
                merged.flagged_items.append(label)

    return merged
