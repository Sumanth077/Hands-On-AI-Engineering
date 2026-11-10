import unittest

from merger import merge_profiles
from schemas import ClinicalProfile, LabFinding


def profile(status: str, value: str) -> ClinicalProfile:
    return ClinicalProfile(
        lab_findings=[
            LabFinding(
                test="Potassium",
                value=value,
                normal_range="3.5–5.0 mmol/L",
                status=status,
            )
        ]
    )


class MergeProfilesTests(unittest.TestCase):
    def test_later_critical_result_replaces_abnormal(self):
        merged = merge_profiles([profile("abnormal", "5.4"), profile("critical", "6.9")])

        self.assertEqual(len(merged.lab_findings), 1)
        self.assertEqual(merged.lab_findings[0].status, "critical")
        self.assertEqual(merged.lab_findings[0].value, "6.9")
        self.assertTrue(any("[critical]" in item for item in merged.flagged_items))

    def test_later_less_severe_result_does_not_replace_critical(self):
        merged = merge_profiles([profile("critical", "6.9"), profile("abnormal", "5.4")])

        self.assertEqual(merged.lab_findings[0].status, "critical")
        self.assertEqual(merged.lab_findings[0].value, "6.9")

    def test_critical_ranks_above_normal_and_abnormal(self):
        merged = merge_profiles(
            [profile("normal", "4.0"), profile("abnormal", "5.4"), profile("critical", "6.9")]
        )

        self.assertEqual(merged.lab_findings[0].status, "critical")
        self.assertEqual(merged.lab_findings[0].value, "6.9")


if __name__ == "__main__":
    unittest.main()
