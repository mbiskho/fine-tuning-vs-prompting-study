import pytest

from ft_vs_prompt.data.generator import (
    DataConfig,
    TaskType,
    _generate_classification_synthetic,
    _generate_extraction_synthetic,
    _generate_style_synthetic,
    CLASSIFICATION_TEMPLATES,
    EXTRACTION_TEMPLATES,
    STYLE_TEMPLATES,
    generate_classification_data,
    generate_extraction_data,
    generate_style_data,
)


class TestDataGenerator:
    def test_classification_data(self):
        config = DataConfig(task=TaskType.CLASSIFICATION, train_sizes=[100, 500], val_size=100, test_size=200)
        ds = generate_classification_data(config)

        assert "test" in ds
        assert "validation" in ds
        assert "train_100" in ds
        assert "train_500" in ds
        assert len(ds["test"]) == 200
        assert len(ds["validation"]) == 100
        assert len(ds["train_100"]) == 100
        assert len(ds["train_500"]) == 500

    def test_extraction_data(self):
        config = DataConfig(task=TaskType.STRUCTURED_EXTRACTION, train_sizes=[100], val_size=100, test_size=200)
        ds = generate_extraction_data(config)

        assert "test" in ds
        assert "validation" in ds
        assert "train_100" in ds
        sample = ds["test"][0]
        assert "instruction" in sample
        assert "input" in sample
        assert "output" in sample

    def test_style_data(self):
        config = DataConfig(task=TaskType.STYLE_CONSTRAINED_GENERATION, train_sizes=[100], val_size=100, test_size=200)
        ds = generate_style_data(config)

        assert "test" in ds
        assert "train_100" in ds

    def test_synthetic_classification(self):
        templates = list(CLASSIFICATION_TEMPLATES.values())
        synthetic = _generate_classification_synthetic(templates, 300, seed=42)
        assert len(synthetic) == 300

    def test_synthetic_extraction(self):
        templates = list(EXTRACTION_TEMPLATES.values())
        synthetic = _generate_extraction_synthetic(templates, 200, seed=42)
        assert len(synthetic) == 200

    def test_synthetic_style(self):
        templates = list(STYLE_TEMPLATES.values())
        synthetic = _generate_style_synthetic(templates, 300, seed=42)
        assert len(synthetic) == 300

    def test_reproducibility(self):
        config1 = DataConfig(task=TaskType.CLASSIFICATION, train_sizes=[100], val_size=50, test_size=100, seed=42)
        config2 = DataConfig(task=TaskType.CLASSIFICATION, train_sizes=[100], val_size=50, test_size=100, seed=42)

        ds1 = generate_classification_data(config1)
        ds2 = generate_classification_data(config2)

        assert ds1["test"][0]["input"] == ds2["test"][0]["input"]

    def test_config_defaults(self):
        config = DataConfig(task=TaskType.CLASSIFICATION)
        assert config.train_sizes == [100, 500, 2000]
        assert config.val_size == 500
        assert config.test_size == 1000
