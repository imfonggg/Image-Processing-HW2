"""
Tests for Image Processing HW2.

Run with:
    python -m pytest test_image_processing.py -v
"""

import numpy as np
import pytest

from histogram import (
    optimal_threshold,
    algorithm_a,
    algorithm_b,
    algorithm_d,
    augment_greyscale,
    rotate_90,
    rotate_180,
    rotate_270,
)
from color_processing import apply_algorithm_a_color, augment_color


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grey():
    """50x50 greyscale image with a gradient from 0 to 255."""
    data = np.tile(np.linspace(0, 255, 50, dtype=np.uint8), (50, 1))
    return data


@pytest.fixture
def uniform_grey():
    """50x50 completely uniform (single value) greyscale image."""
    return np.full((50, 50), 128, dtype=np.uint8)


@pytest.fixture
def simple_color():
    """50x50 RGB image with gradient in red channel only."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 50, dtype=np.uint8), (50, 1))
    return img


# ---------------------------------------------------------------------------
# Optimal Thresholding
# ---------------------------------------------------------------------------

class TestOptimalThreshold:
    def test_returns_float(self, simple_grey):
        T = optimal_threshold(simple_grey)
        assert isinstance(T, float)

    def test_within_range(self, simple_grey):
        T = optimal_threshold(simple_grey)
        assert 0 <= T <= 255

    def test_uniform_image(self, uniform_grey):
        # All pixels same value: threshold must still be a float in [0,255]
        T = optimal_threshold(uniform_grey)
        assert 0 <= T <= 255

    def test_two_class_image(self):
        # Image with two distinct clusters: 0 and 200
        img = np.array([[0, 0, 200, 200],
                        [0, 0, 200, 200]], dtype=np.uint8)
        T = optimal_threshold(img)
        # T should lie between the two means: 0 < T < 200
        assert 0 < T < 200


# ---------------------------------------------------------------------------
# Algorithm A
# ---------------------------------------------------------------------------

class TestAlgorithmA:
    def test_output_shape(self, simple_grey):
        result = algorithm_a(simple_grey)
        assert result.shape == simple_grey.shape

    def test_output_dtype(self, simple_grey):
        result = algorithm_a(simple_grey)
        assert result.dtype == np.uint8

    def test_output_range(self, simple_grey):
        result = algorithm_a(simple_grey)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bright_pixels_unchanged(self):
        """Pixels >= T must not be modified."""
        img = np.array([[0, 50, 200, 250],
                        [10, 60, 210, 240]], dtype=np.uint8)
        T = optimal_threshold(img)
        result = algorithm_a(img)
        bright_mask = img >= T
        np.testing.assert_array_equal(result[bright_mask], img[bright_mask])

    def test_uniform_image_no_crash(self, uniform_grey, capsys):
        """Uniform image has no dark pixels after thresholding — no crash."""
        result = algorithm_a(uniform_grey)
        assert result.shape == uniform_grey.shape
        # Should print a message about no dark pixels or degenerate range
        out = capsys.readouterr().out
        assert len(out) > 0 or np.array_equal(result, uniform_grey)

    def test_no_dark_pixels_message(self, capsys):
        """All-bright image should trigger the 'no dark pixels' message."""
        img = np.full((20, 20), 250, dtype=np.uint8)
        result = algorithm_a(img)
        out = capsys.readouterr().out
        # Either unchanged or a message was printed
        assert np.array_equal(result, img) or "[Algorithm A]" in out


# ---------------------------------------------------------------------------
# Algorithm B
# ---------------------------------------------------------------------------

class TestAlgorithmB:
    def test_output_shape(self, simple_grey):
        result = algorithm_b(simple_grey)
        assert result.shape == simple_grey.shape

    def test_output_dtype(self, simple_grey):
        result = algorithm_b(simple_grey)
        assert result.dtype == np.uint8

    def test_output_range(self, simple_grey):
        result = algorithm_b(simple_grey)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_odd_dimensions(self):
        """Odd-dimension images must be handled without error."""
        img = np.random.randint(0, 200, (51, 53), dtype=np.uint8)
        result = algorithm_b(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

class TestRotations:
    def test_rotate_90_shape(self, simple_grey):
        r = rotate_90(simple_grey)
        # 90° rotation swaps H and W
        assert r.shape == (simple_grey.shape[1], simple_grey.shape[0])

    def test_rotate_180_shape(self, simple_grey):
        r = rotate_180(simple_grey)
        assert r.shape == simple_grey.shape

    def test_rotate_270_shape(self, simple_grey):
        r = rotate_270(simple_grey)
        assert r.shape == (simple_grey.shape[1], simple_grey.shape[0])

    def test_four_rotations_identity(self, simple_grey):
        """Four 90° rotations should return to the original."""
        r = rotate_90(rotate_90(rotate_90(rotate_90(simple_grey))))
        np.testing.assert_array_equal(r, simple_grey)


# ---------------------------------------------------------------------------
# 1C: Grey augmentation
# ---------------------------------------------------------------------------

class TestAugmentGreyscale:
    def test_returns_12_images(self, simple_grey):
        images = augment_greyscale(simple_grey)
        assert len(images) == 12

    def test_all_are_tuples(self, simple_grey):
        images = augment_greyscale(simple_grey)
        for item in images:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_all_uint8(self, simple_grey):
        images = augment_greyscale(simple_grey)
        for arr, _ in images:
            assert arr.dtype == np.uint8

    def test_labels_are_strings(self, simple_grey):
        images = augment_greyscale(simple_grey)
        for _, label in images:
            assert isinstance(label, str)

    def test_first_image_is_original(self, simple_grey):
        images = augment_greyscale(simple_grey)
        np.testing.assert_array_equal(images[0][0], simple_grey)


# ---------------------------------------------------------------------------
# Algorithm D (1D extra credit)
# ---------------------------------------------------------------------------

class TestAlgorithmD:
    def test_output_shape(self, simple_grey):
        result = algorithm_d(simple_grey, p_percent=10.0)
        assert result.shape == simple_grey.shape

    def test_output_dtype(self, simple_grey):
        result = algorithm_d(simple_grey, p_percent=10.0)
        assert result.dtype == np.uint8

    def test_output_range(self, simple_grey):
        result = algorithm_d(simple_grey, p_percent=10.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_small_region_stops_recursion(self):
        """Region smaller than 10x10 should not recurse indefinitely."""
        img = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        result = algorithm_d(img, p_percent=5.0)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# 3A: Color histogram modification
# ---------------------------------------------------------------------------

class TestApplyAlgorithmAColor:
    def test_output_shape(self, simple_color):
        result = apply_algorithm_a_color(simple_color, "all")
        assert result.shape == simple_color.shape

    def test_output_dtype(self, simple_color):
        result = apply_algorithm_a_color(simple_color, "all")
        assert result.dtype == np.uint8

    def test_single_channel_r(self, simple_color):
        result = apply_algorithm_a_color(simple_color, "R")
        # Channels 1 and 2 (G, B) must be unchanged
        np.testing.assert_array_equal(result[:, :, 1], simple_color[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], simple_color[:, :, 2])

    def test_single_channel_g(self, simple_color):
        result = apply_algorithm_a_color(simple_color, "G")
        np.testing.assert_array_equal(result[:, :, 0], simple_color[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 2], simple_color[:, :, 2])

    def test_single_channel_b(self, simple_color):
        result = apply_algorithm_a_color(simple_color, "B")
        np.testing.assert_array_equal(result[:, :, 0], simple_color[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], simple_color[:, :, 1])

    def test_invalid_channel_raises(self, simple_color):
        with pytest.raises(ValueError):
            apply_algorithm_a_color(simple_color, "X")

    def test_output_range(self, simple_color):
        result = apply_algorithm_a_color(simple_color, "all")
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# 3B: Color augmentation
# ---------------------------------------------------------------------------

class TestAugmentColor:
    def test_returns_20_images(self, simple_color):
        images = augment_color(simple_color)
        assert len(images) == 20

    def test_all_are_tuples(self, simple_color):
        images = augment_color(simple_color)
        for item in images:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_all_uint8(self, simple_color):
        images = augment_color(simple_color)
        for arr, _ in images:
            assert arr.dtype == np.uint8

    def test_labels_are_strings(self, simple_color):
        images = augment_color(simple_color)
        for _, label in images:
            assert isinstance(label, str)

    def test_first_image_is_original(self, simple_color):
        images = augment_color(simple_color)
        np.testing.assert_array_equal(images[0][0], simple_color)

    def test_output_range(self, simple_color):
        images = augment_color(simple_color)
        for arr, _ in images:
            assert arr.min() >= 0
            assert arr.max() <= 255
