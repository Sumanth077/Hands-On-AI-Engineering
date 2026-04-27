from pydantic import BaseModel, Field
from typing import Optional, List


class Medication(BaseModel):
    drug_name: str = Field(description="Name of the drug/medication as written on prescription")
    dosage: Optional[str] = Field(default=None, description="Dosage amount and unit, e.g. '500mg', '10mg/5ml'")
    frequency: Optional[str] = Field(default=None, description="How often to take, e.g. 'twice daily', 'every 8 hours', 'TID'")
    duration: Optional[str] = Field(default=None, description="Duration of treatment, e.g. '7 days', '2 weeks'")
    is_validated: bool = Field(default=False, description="Whether drug name was validated against RxNorm")
    validation_note: Optional[str] = Field(default=None, description="Note about validation result")


class Prescription(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Full name of the patient")
    doctor_name: Optional[str] = Field(default=None, description="Name of the prescribing doctor")
    date: Optional[str] = Field(default=None, description="Date on the prescription")
    medications: List[Medication] = Field(description="List of all medications prescribed")
    notes: Optional[str] = Field(default=None, description="Any additional notes, instructions, or diagnoses on the prescription")
    illegible_fields: List[str] = Field(
        default=[],
        description="List of fields or sections that were illegible or unclear, e.g. ['patient address', 'drug 2 dosage']"
    )
