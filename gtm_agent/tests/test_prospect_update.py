import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent import gtm_agent


class ProspectUpdateTest(unittest.TestCase):
    def test_prospect_update_refreshes_profile_and_score(self):
        prospect_id = "LEAD-71001"
        technology = "Kafka"
        record = data_service.PROSPECTS[prospect_id]
        original_stack = record["tech_stack"]
        original_profile = data_service._PROFILES.pop(prospect_id, None)

        try:
            data_service._PROFILES[prospect_id] = {
                "prospect_id": prospect_id,
                "tech_stack": list(original_stack),
            }
            update = data_service.update_prospect_info(prospect_id, technology)
            profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})

            captured = {}

            class FakeScorer:
                def invoke(self, messages):
                    captured["messages"] = messages
                    return SimpleNamespace(model_dump=lambda: {"score": 80})

            original_scorer = gtm_agent._scoring_llm
            gtm_agent._scoring_llm = FakeScorer()
            try:
                gtm_agent.score_prospect.invoke({
                    "prospect_profile": profile["prospect_profile"],
                    "offering": data_service.get_offering("OFFER-10005"),
                })
            finally:
                gtm_agent._scoring_llm = original_scorer

            self.assertEqual(update["tech_stack"][-1], technology)
            self.assertEqual(profile["prospect_profile"]["tech_stack"][-1], technology)
            self.assertIn(technology, data_service.fetch_tech_stack(prospect_id))
            self.assertIn(technology, captured["messages"][1]["content"])
        finally:
            record["tech_stack"] = original_stack
            if original_profile is None:
                data_service._PROFILES.pop(prospect_id, None)
            else:
                data_service._PROFILES[prospect_id] = original_profile
