import pytest

from ft_vs_prompt.tco.model import (
    CrossoverAnalysis,
    TCOResult,
    compute_finetuning_tco,
    compute_prompting_tco,
    find_crossover,
    generate_decision_guide,
)


class TestTCO:
    def test_prompting_tco(self):
        tco = compute_prompting_tco(tokens_per_request=500)
        assert tco.method == "few_shot_prompting"
        assert tco.upfront_cost == 0.0
        assert tco.monthly_fixed_cost == 0.0
        assert tco.cost_per_1k_requests > 0
        assert tco.cost_at_10k_daily > tco.cost_at_1k_daily

    def test_finetuning_tco(self):
        tco = compute_finetuning_tco(
            gpu_hours=4.0,
            gpu_cost_per_hour=1.50,
            dataset_labour_hours=20.0,
            labour_rate_per_hour=75.0,
        )
        assert tco.method == "lora_fine_tuning"
        assert tco.upfront_cost > 0
        assert tco.monthly_fixed_cost > 0

    def test_finetuning_tco_upfront(self):
        tco = compute_finetuning_tco(
            gpu_hours=4.0,
            gpu_cost_per_hour=1.50,
            dataset_labour_hours=20.0,
            labour_rate_per_hour=75.0,
        )
        expected_dataset = 20.0 * 75.0
        expected_training = 4.0 * 1.50
        assert tco.upfront_cost == round(expected_dataset + expected_training, 2)

    def test_crossover_exists(self):
        prompting = compute_prompting_tco(tokens_per_request=500)
        finetuning = compute_finetuning_tco()
        analysis = find_crossover(prompting, finetuning, "classification")

        assert analysis.crossover_volume_daily > 0
        assert analysis.recommendation != ""

    def test_crossover_high_volume(self):
        prompting = compute_prompting_tco(tokens_per_request=100)
        finetuning = compute_finetuning_tco(serving_gpu_cost_per_hour=0.10)
        analysis = find_crossover(prompting, finetuning, "test")

        assert analysis.crossover_volume_daily > 0

    def test_decision_guide_generation(self):
        analyses = [
            find_crossover(compute_prompting_tco(), compute_finetuning_tco(), "classification"),
            find_crossover(compute_prompting_tco(), compute_finetuning_tco(), "extraction"),
        ]
        guide = generate_decision_guide(analyses)

        assert "Decision Guide" in guide or "decision" in guide.lower()
        assert "classification" in guide.lower()
        assert "fine-tuning" in guide.lower()

    def test_tco_notes(self):
        tco = compute_finetuning_tco()
        assert len(tco.notes) > 0
        assert any("dataset" in n.lower() for n in tco.notes)
        assert any("training" in n.lower() for n in tco.notes)
