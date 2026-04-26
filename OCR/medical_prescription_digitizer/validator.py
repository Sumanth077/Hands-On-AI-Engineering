import requests
from schemas import Medication, Prescription

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
REQUEST_TIMEOUT = 10


def validate_drug_name(drug_name: str) -> tuple[bool, str | None]:
    """
    Query RxNorm API to validate a drug name.
    Returns (is_valid, rxcui_or_none).
    """
    try:
        response = requests.get(
            RXNORM_BASE_URL,
            params={"name": drug_name},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId", [])

        if rxnorm_ids:
            return True, rxnorm_ids[0]
        return False, None

    except requests.exceptions.Timeout:
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def validate_prescription_drugs(prescription: Prescription) -> Prescription:
    """Validate all drug names in a prescription against RxNorm and update in-place."""
    for med in prescription.medications:
        if not med.drug_name or med.drug_name.strip() == "":
            med.is_validated = False
            med.validation_note = "Drug name is empty or illegible"
            continue

        is_valid, rxcui = validate_drug_name(med.drug_name.strip())

        if is_valid:
            med.is_validated = True
            med.validation_note = f"Validated — RxNorm ID: {rxcui}"
        else:
            med.is_validated = False
            med.validation_note = "Drug name not found in RxNorm — possible misread"

    return prescription
