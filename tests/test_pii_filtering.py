import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import (
    build_prospect_profile,
    get_prospect,
    score_prospect,
)
from gtm_agent.gtm_records import OFFERINGS


SENSITIVE_KEYS = {
    "billing_qualification",
    "tax_id",
    "date_of_birth",
    "card_on_file",
    "credit_check_ref",
}


class _ScoreResult:
    def model_dump(self):
        return {"score": 80}


class _CapturingLlm:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _ScoreResult()


class PiiFilteringTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def test_prospect_tools_and_cache_exclude_billing_data(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-50005"})
        profile = build_prospect_profile.invoke({"prospect_id": "LEAD-90001"})

        self.assertTrue(SENSITIVE_KEYS.isdisjoint(prospect["prospect"]))
        self.assertTrue(SENSITIVE_KEYS.isdisjoint(profile["prospect_profile"]))

        cached = data_service.get_profile_from_db("LEAD-90001")["prospect_profile"]
        self.assertTrue(SENSITIVE_KEYS.isdisjoint(cached))

        profile["prospect_profile"]["engagement_history"].clear()
        cached_again = data_service.get_profile_from_db("LEAD-90001")["prospect_profile"]
        self.assertTrue(cached_again["engagement_history"])

    def test_score_payload_ignores_extra_profile_fields(self):
        llm = _CapturingLlm()
        profile = {
            "prospect_id": "LEAD-50005",
            "name": "Test Prospect",
            "annual_revenue": 1000000,
            "tech_stack": ["AWS"],
            "account_details": [],
            "engagement_history": [],
            "billing_qualification": {
                "tax_id": "123",
                "date_of_birth": "2000-01-01",
                "card_on_file": "4111111111111111",
                "credit_check_ref": "EXPN-1",
            },
            "unexpected_field": "must not be serialized",
        }

        with patch("gtm_agent.gtm_agent._scoring_llm", llm):
            score_prospect.invoke({
                "prospect_profile": profile,
                "offering": OFFERINGS["OFFER-10001"],
            })

        user_payload = json.loads(llm.messages[1]["content"].split(
            "\n\nProspect profile:\n", 1
        )[1])
        self.assertEqual(
            set(user_payload),
            {
                "prospect_id",
                "name",
                "annual_revenue",
                "tech_stack",
                "account_details",
                "engagement_history",
            },
        )
        self.assertTrue(SENSITIVE_KEYS.isdisjoint(user_payload))


if __name__ == "__main__":
    unittest.main()
