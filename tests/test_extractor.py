"""Tests for the keyword-based metadata extractor."""

from __future__ import annotations

import pytest
from zotero_extract.extractor import extract


class TestExtract:
    # ------------------------------------------------------------------
    # task_type
    # ------------------------------------------------------------------
    def test_takeover_request(self):
        text = "Participants were asked to perform a takeover request after 30 s."
        result = extract(text)
        assert result["task_type"] == "takeover_request"

    def test_lane_keeping(self):
        text = "The experiment focused on lane keeping and steering wheel corrections."
        result = extract(text)
        assert result["task_type"] == "lane_keeping"

    def test_hazard_detection(self):
        text = "A hazard detection task was used to assess forward collision warning."
        result = extract(text)
        assert result["task_type"] == "hazard_detection"

    def test_car_following(self):
        text = "The car following behaviour was measured using time headway."
        result = extract(text)
        assert result["task_type"] == "car_following"

    # ------------------------------------------------------------------
    # driving_scenario
    # ------------------------------------------------------------------
    def test_highway_scenario(self):
        text = "Participants drove on a simulated highway at 120 km/h."
        result = extract(text)
        assert result["driving_scenario"] == "highway"

    def test_urban_scenario(self):
        text = "The urban road network included traffic lights and pedestrians."
        result = extract(text)
        assert result["driving_scenario"] == "urban"

    def test_simulator_scenario(self):
        text = "A high-fidelity driving simulator was used for the study."
        result = extract(text)
        assert result["driving_scenario"] == "simulator"

    # ------------------------------------------------------------------
    # level_of_automation
    # ------------------------------------------------------------------
    def test_sae_l2(self):
        text = "The vehicle operated at SAE Level 2, providing partial automation."
        result = extract(text)
        assert result["level_of_automation"] == "SAE_L2"

    def test_sae_l3(self):
        text = "Conditional automation (SAE level 3) was enabled during the trial."
        result = extract(text)
        assert result["level_of_automation"] == "SAE_L3"

    def test_adas(self):
        text = "Advanced driver assistance systems (ADAS) supported the driver."
        result = extract(text)
        assert result["level_of_automation"] == "assisted"

    # ------------------------------------------------------------------
    # visual_cue
    # ------------------------------------------------------------------
    def test_hud(self):
        text = "Warnings were presented on a head-up display (HUD)."
        result = extract(text)
        assert result["visual_cue"] == "HUD"

    def test_ar_display(self):
        text = "An augmented reality overlay was projected onto the windshield."
        result = extract(text)
        assert result["visual_cue"] == "AR_display"

    def test_ambient_light(self):
        text = "Ambient light cues provided peripheral colour-coded warnings."
        result = extract(text)
        assert result["visual_cue"] == "ambient_light"

    def test_auditory(self):
        text = "An auditory beep was used to alert the driver."
        result = extract(text)
        assert result["visual_cue"] == "auditory"

    # ------------------------------------------------------------------
    # confidence
    # ------------------------------------------------------------------
    def test_confidence_range(self):
        text = "Some unrelated text about cooking and recipes."
        result = extract(text)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_high_for_rich_text(self):
        text = (
            "Participants performed a takeover request on a simulated highway. "
            "The vehicle was operating at SAE Level 2. "
            "Warnings were shown on a head-up display (HUD)."
        )
        result = extract(text)
        assert result["confidence"] > 0.3

    # ------------------------------------------------------------------
    # abstract augmentation
    # ------------------------------------------------------------------
    def test_abstract_provides_signal(self):
        result_no_abstract = extract("Some generic text.")
        result_with_abstract = extract(
            "Some generic text.",
            abstract="Participants performed a lane keeping task on the highway.",
        )
        assert result_with_abstract["confidence"] >= result_no_abstract["confidence"]
