import pytest
from click.testing import CliRunner

from ft_vs_prompt.cli import main


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Fine-Tuning" in result.output

    def test_generate_help(self, runner):
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0

    def test_generate_small(self, runner):
        with runner.isolated_filesystem():
            result = runner.invoke(main, [
                "generate", "--output", "data", "--train-sizes", "50,100",
            ])
            assert result.exit_code == 0

    def test_evaluate_help(self, runner):
        result = runner.invoke(main, ["evaluate-prompting", "--help"])
        assert result.exit_code == 0

    def test_crossover_help(self, runner):
        result = runner.invoke(main, ["crossover", "--help"])
        assert result.exit_code == 0

    def test_crossover(self, runner):
        result = runner.invoke(main, ["crossover", "--task", "classification"])
        assert result.exit_code == 0
        assert "Crossover volume" in result.output

    def test_decision_guide(self, runner):
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["decision-guide", "--output", "docs"])
            assert result.exit_code == 0

    def test_train_help(self, runner):
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "--base-model" in result.output
