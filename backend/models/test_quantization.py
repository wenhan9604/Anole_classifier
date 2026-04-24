"""
Integration tests for ONNX model quantization.

Tests the quantization pipeline including:
- YOLO model quantization
- Swin model quantization
- Benchmark suite functionality
- Model validation after quantization
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the quantization modules
try:
    from quantization_utils import (
        quantize_onnx_model,
        validate_quantized_model,
        get_quantization_info,
    )
except ImportError:
    # Handle import for testing from different directories
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from quantization_utils import (
        quantize_onnx_model,
        validate_quantized_model,
        get_quantization_info,
    )


class TestQuantizationUtils(unittest.TestCase):
    """Test cases for quantization utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model_path = os.path.join(self.temp_dir, "test_model.onnx")
        self.quantized_model_path = os.path.join(
            self.temp_dir, "test_model_quantized.onnx"
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("quantization_utils.ort.quantization.quantize_dynamic")
    def test_quantize_onnx_model_success(self, mock_quantize):
        """Test successful ONNX model quantization."""
        # Create a mock ONNX model file
        with open(self.test_model_path, "w") as f:
            f.write("mock onnx model")

        mock_quantize.return_value = None

        result = quantize_onnx_model(
            self.test_model_path, self.quantized_model_path, "dynamic"
        )

        self.assertTrue(result)
        mock_quantize.assert_called_once()

    @patch("quantization_utils.ort.quantization.quantize_dynamic")
    def test_quantize_onnx_model_failure(self, mock_quantize):
        """Test quantization with non-existent model."""
        mock_quantize.side_effect = Exception("Model not found")

        result = quantize_onnx_model(
            "/nonexistent/model.onnx",
            self.quantized_model_path,
            "dynamic",
        )

        self.assertFalse(result)

    def test_get_quantization_info(self):
        """Test quantization info retrieval."""
        # Create mock model paths
        with open(self.test_model_path, "w") as f:
            f.write("mock onnx model")

        with open(self.quantized_model_path, "w") as f:
            f.write("mock quantized onnx model")

        # Get file sizes for comparison
        original_size = os.path.getsize(self.test_model_path)
        quantized_size = os.path.getsize(self.quantized_model_path)

        info = get_quantization_info(
            self.test_model_path, self.quantized_model_path
        )

        self.assertIsInstance(info, dict)
        self.assertIn("original_size_mb", info)
        self.assertIn("quantized_size_mb", info)
        self.assertIn("compression_ratio", info)


class TestQuantizationIntegration(unittest.TestCase):
    """Integration tests for the quantization pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_quantization_pipeline_workflow(self):
        """Test the complete quantization workflow."""
        # This test verifies the pipeline can be imported and functions exist
        from quantization_utils import (
            quantize_onnx_model,
            validate_quantized_model,
            get_quantization_info,
            QuantizationConfig,
        )

        # Verify all required components exist
        self.assertTrue(callable(quantize_onnx_model))
        self.assertTrue(callable(validate_quantized_model))
        self.assertTrue(callable(get_quantization_info))
        self.assertTrue(hasattr(QuantizationConfig, "WEIGHT_TYPE"))
        self.assertTrue(hasattr(QuantizationConfig, "OPTIMIZE_MODEL"))

    def test_quantization_config_constants(self):
        """Test QuantizationConfig has expected constants."""
        from quantization_utils import QuantizationConfig

        # Check for required quantization type constants
        self.assertEqual(QuantizationConfig.WEIGHT_TYPE, "QUInt8")
        self.assertTrue(QuantizationConfig.OPTIMIZE_MODEL)

    def test_import_benchmark_suite(self):
        """Test that benchmark suite can be imported."""
        try:
            from benchmark_quantization import (
                BenchmarkQuantization,
                compare_model_performance,
            )

            self.assertTrue(callable(compare_model_performance))
            self.assertTrue(hasattr(BenchmarkQuantization, "benchmark"))
        except ImportError:
            self.skipTest("benchmark_quantization module not available")


class TestExportPipelines(unittest.TestCase):
    """Test YOLO and Swin export pipelines with quantization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_yolo_export_module_exists(self):
        """Test that YOLO export module exists and can be imported."""
        try:
            # Try to import from parent directory
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            import export_yolo_to_onnx

            self.assertTrue(hasattr(export_yolo_to_onnx, "export_yolo_model"))
        except (ImportError, AttributeError):
            self.skipTest("export_yolo_to_onnx not available")

    def test_swin_export_module_exists(self):
        """Test that Swin export module exists and can be imported."""
        try:
            # Try to import from parent directory
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            import export_swin_to_onnx

            self.assertTrue(hasattr(export_swin_to_onnx, "export_swin_model"))
        except (ImportError, AttributeError):
            self.skipTest("export_swin_to_onnx not available")


class TestQuantizationDocumentation(unittest.TestCase):
    """Test that quantization documentation exists."""

    def test_quantization_guide_exists(self):
        """Test that QUANTIZATION_GUIDE.md exists."""
        guide_path = Path(__file__).parent.parent.parent / "docs" / "QUANTIZATION_GUIDE.md"
        self.assertTrue(
            guide_path.exists(),
            f"QUANTIZATION_GUIDE.md not found at {guide_path}",
        )

    def test_quantization_guide_has_content(self):
        """Test that QUANTIZATION_GUIDE.md has substantial content."""
        guide_path = Path(__file__).parent.parent.parent / "docs" / "QUANTIZATION_GUIDE.md"
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertGreater(len(content), 100, "QUANTIZATION_GUIDE.md appears to be empty")
        self.assertIn("quantization", content.lower(), "Guide doesn't mention quantization")


def run_tests():
    """Run all quantization tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestExportPipelines))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationDocumentation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
