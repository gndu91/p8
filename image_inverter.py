"""
Based on Pycharm Community's inversion algorithm (written in Kotlin)

"""
import io
import colorsys
from os import urandom, PathLike
from pathlib import Path
from time import time_ns
from typing import Tuple, List, Optional, IO, Any

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.figure import Figure

# Type aliases for color tuples for clarity
ColorTuple = Tuple[int, int, int]
ColorTupleA = Tuple[int, int, int, int]


class ImageInverter:
    """
    A Python implementation of JetBrains' ImageInverter class for inverting
    images to fit dark themes. It intelligently decides whether to invert an
    image and applies a color-preserving inversion based on theme-specific
    foreground and background colors.
    """
    # Constants for HSL/RGB array indices, mirroring the original class
    _H, _S, _L = 0, 1, 2
    _R, _G, _B = 0, 1, 2

    def __init__(self,
                 foreground: Optional[ColorTuple] = None,
                 background: Optional[ColorTuple] = None):
        """
        Initializes the inverter with the theme's foreground and background colors.
        These are used to correctly map the luminance range of the inverted image.

        The defaults are based on the standard JetBrains Darcula theme, which
        is the most likely context for this code.

        Args:
            foreground: An (r, g, b) tuple for the theme's primary text color.
                        This is the color that black elements from the original
                        image (like lines or text) will be mapped to.
                        Defaults to Darcula's text color: (169, 183, 198).
            background: An (r, g, b) tuple for the theme's primary background color.
                        This is the color that white elements from the original
                        image (like a plot's background) will be mapped to.
                        Defaults to Darcula's background: (43, 43, 43).
        """
        # Use Darcula theme colors as the default "best guess"
        if foreground is None:
            foreground = (169, 183, 198)  # Darcula foreground: #A9B7C6
        if background is None:
            background = (43, 43, 43)     # Darcula background: #2B2B2B

        self.rgb = [0.0, 0.0, 0.0]
        self.hsl = [0.0, 0.0, 0.0]

        # Get HSL of the theme's "white" (foreground)
        self.white_hsl = self._convert_rgb_to_hsl(foreground + (255,))
        # Get HSL of the theme's "black" (background)
        self.black_hsl = self._convert_rgb_to_hsl(background + (255,))

    def should_invert(self, image: Image.Image, brightness_threshold: float = 0.65) -> bool:
        """
        Checks if an image should be inverted for a dark theme.

        Inversion is recommended if the image has high average brightness and is
        not overly complex (e.g., a data plot), or if it has a clearly
        identifiable light-colored background.

        Args:
            image: The PIL.Image object to analyze.
            brightness_threshold: Images with average brightness exceeding this
                                  will be considered for inversion.

        Returns:
            True if it's recommended to invert the image, False otherwise.
        """
        # Ensure image has an alpha channel for consistent processing
        img_rgba = image.convert('RGBA')
        colors_sample = self._get_image_sample(img_rgba)

        if colors_sample.shape[0] == 0:
            return False

        num_pixels = colors_sample.shape[0]

        # An image is "complex" if it has many unique colors
        num_colors_in_complex_image = 5000
        num_colors_threshold = min(num_pixels / 3, num_colors_in_complex_image)

        brightness_values = np.apply_along_axis(self._get_brightness, 1, colors_sample)
        average_brightness = np.sum(brightness_values) / num_pixels

        unique_colors = np.unique(colors_sample, axis=0)
        num_colors = unique_colors.shape[0]

        # Condition 1: Bright and not complex
        if average_brightness > brightness_threshold and num_colors < num_colors_threshold:
            return True

        # Condition 2: Has a dominant light background
        if self._has_light_background(colors_sample, brightness_threshold):
            return True

        return False

    def invert_color(self, color: ColorTupleA) -> ColorTupleA:
        """Inverts a single RGBA color tuple."""
        hsl_result, alpha_float = self._invert_pixel_core(color)
        return self._convert_hsl_to_rgb(hsl_result, alpha_float)

    def invert_image(self, image: Image.Image) -> Image.Image:
        """
        Inverts a PIL Image.

        This method handles both palette-based ('P' mode) and standard RGB/RGBA
        images efficiently.

        Args:
            image: The PIL.Image object to invert.

        Returns:
            A new PIL.Image object with inverted colors.
        """
        if image.mode == 'P':
            return self._invert_palette_image(image)
        else:
            return self._invert_rgba_image_fast(image)

    def invert_image_bytes(self, content: bytes) -> bytes:
        """
        Reads an image from bytes, inverts it, and returns the inverted image as bytes.

        Args:
            content: A byte string containing the image data (e.g., PNG, JPEG).

        Returns:
            A byte string of the inverted image in PNG format.
        """
        try:
            image = Image.open(io.BytesIO(content))
        except Exception:
            # If the content is not a valid image, return it as is
            return content

        inverted_image = self.invert_image(image)

        output = io.BytesIO()
        inverted_image.save(output, format='PNG')
        return output.getvalue()

    def _get_image_sample(self, image: Image.Image) -> np.ndarray:
        """
        Gets a sample of pixels from the image for color analysis.

        For small images, all pixels are returned. For larger images, samples are
        taken from the four corners and the center. This is much faster than
        analyzing every pixel.
        """
        if image.height < 10 or image.width < 10:
            return np.array(image).reshape(-1, 4)

        img_arr = np.array(image)

        default_spot_size = min(
            max(image.height // 10, image.width // 10),
            min(image.height, image.width)
        )
        spot_h = min(image.height, default_spot_size)
        spot_w = min(image.width, default_spot_size)

        # Snap central spot coordinates to a grid based on spot size
        center_x = (image.width // 2 // spot_w) * spot_w
        center_y = (image.height // 2 // spot_h) * spot_h

        # Extract corner and center spots
        top_left = img_arr[0:spot_h, 0:spot_w]
        top_right = img_arr[0:spot_h, -spot_w:]
        bottom_left = img_arr[-spot_h:, 0:spot_w]
        bottom_right = img_arr[-spot_h:, -spot_w:]
        center = img_arr[center_y:center_y + spot_h, center_x:center_x + spot_w]

        # Concatenate spots and reshape to a list of pixels (N, 4)
        sample = np.concatenate([
            top_left.reshape(-1, 4),
            top_right.reshape(-1, 4),
            bottom_left.reshape(-1, 4),
            bottom_right.reshape(-1, 4),
            center.reshape(-1, 4)
        ])
        return sample

    def _get_brightness(self, color_tuple: ColorTupleA) -> float:
        """Calculates brightness (Value from HSV) for a single RGBA color."""
        r, g, b, _ = color_tuple
        # colorsys expects float values in [0, 1] range
        _, _, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return v

    def _has_light_background(self, colors: np.ndarray, brightness_threshold: float) -> Optional[bool]:
        """
        Tries to guess whether the image has a dominant light background.
        The background is defined as >50% of pixels having the same light color.
        """
        if colors.shape[0] == 0:
            return None

        unique_rows, counts = np.unique(colors, axis=0, return_counts=True)

        dominant_color_idx = np.argmax(counts)
        dominant_color = unique_rows[dominant_color_idx]
        dominant_pixel_count = counts[dominant_color_idx]

        is_dominant = (dominant_pixel_count / colors.shape[0]) > 0.5
        is_light = self._get_brightness(dominant_color) > brightness_threshold

        return is_dominant and is_light

    def _invert_pixel_core(self, argb_tuple: ColorTupleA) -> Tuple[List[float], float]:
        """
        The core inversion logic. Converts RGBA to HSL, applies the inversion
        formulas, and returns the new HSL and original alpha.
        """
        # Convert to HSL with ranges H[0,360], S[0,100], L[0,100]
        hsl = self._convert_rgb_to_hsl(argb_tuple)

        # Adjust saturation based on theme foreground saturation
        hsl[self._S] = hsl[self._S] * (50.0 + self.white_hsl[self._S]) / 1.5 / 100.0

        # Invert luminance and remap it to the theme's luminance range
        inverted_l = 100.0 - hsl[self._L]
        l_range = self.white_hsl[self._L] - self.black_hsl[self._L]
        hsl[self._L] = (inverted_l * l_range / 100.0) + self.black_hsl[self._L]

        alpha_float = argb_tuple[3] / 255.0
        return hsl, alpha_float

    def _invert_rgba_image_fast(self, image: Image.Image) -> Image.Image:
        """Efficiently inverts an image using NumPy vectorization."""
        img_rgba = image.convert('RGBA')
        data = np.array(img_rgba)

        # Find all unique colors in the image. This is much faster than looping
        # over every pixel, especially for images with limited color palettes.
        unique_colors, inverse_indices = np.unique(
            data.reshape(-1, 4), axis=0, return_inverse=True
        )

        # Create a lookup table (LUT) by inverting each unique color
        inverted_unique_colors = np.array([
            self.invert_color(tuple(color)) for color in unique_colors
        ], dtype=np.uint8)

        # Build the new image data by applying the LUT
        new_data = inverted_unique_colors[inverse_indices].reshape(data.shape)

        return Image.fromarray(new_data, 'RGBA')

    def _invert_palette_image(self, image: Image.Image) -> Image.Image:
        """
        Inverts a palette-based image by inverting its color palette directly.
        This is significantly more efficient than pixel-by-pixel processing.
        """
        if image.mode != 'P':
            return image

        output_image = image.copy()
        palette_data = output_image.getpalette()

        if not palette_data:
            # Fallback for palette image with no actual palette data
            return self._invert_rgba_image_fast(image)

        new_palette = []
        # The palette is a flat list of [R1, G1, B1, R2, G2, B2, ...].
        # We process it in chunks of 3.
        for i in range(0, len(palette_data), 3):
            if i + 3 > len(palette_data):
                new_palette.extend(palette_data[i:])
                break

            r, g, b = palette_data[i:i+3]
            # invert_color expects RGBA and returns RGBA. We use full alpha.
            inv_r, inv_g, inv_b, _ = self.invert_color((r, g, b, 255))
            new_palette.extend([inv_r, inv_g, inv_b])

        output_image.putpalette(new_palette)
        return output_image

    def _convert_rgb_to_hsl(self, rgb_tuple: ColorTupleA) -> List[float]:
        """Converts RGB [0-255] to HSL (H[0-360], S[0-100], L[0-100])."""
        r, g, b, _ = rgb_tuple
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        c_max = max(r, g, b)
        c_min = min(r, g, b)
        delta = c_max - c_min

        l = (c_max + c_min) / 2.0

        if delta == 0:
            s = 0.0
            h = 0.0
        else:
            s = delta / (1 - abs(2 * l - 1))
            if c_max == r:
                h = ((g - b) / delta) % 6
            elif c_max == g:
                h = ((b - r) / delta) + 2
            else: # c_max == b
                h = ((r - g) / delta) + 4

        h = h * 60
        if h < 0:
            h += 360

        return [h, s * 100, l * 100]

    def _convert_hsl_to_rgb(self, hsl: List[float], alpha: float) -> ColorTupleA:
        """Converts HSL (H[0-360], S[0-100], L[0-100]) to RGBA [0-255]."""
        h, s, l = hsl[0], hsl[1] / 100.0, hsl[2] / 100.0

        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs(((h / 60) % 2) - 1))
        m = l - c / 2

        if 0 <= h < 60: r_prime, g_prime, b_prime = c, x, 0
        elif 60 <= h < 120: r_prime, g_prime, b_prime = x, c, 0
        elif 120 <= h < 180: r_prime, g_prime, b_prime = 0, c, x
        elif 180 <= h < 240: r_prime, g_prime, b_prime = 0, x, c
        elif 240 <= h < 300: r_prime, g_prime, b_prime = x, 0, c
        else: r_prime, g_prime, b_prime = c, 0, x

        r = int(round((r_prime + m) * 255))
        g = int(round((g_prime + m) * 255))
        b = int(round((b_prime + m) * 255))
        a = int(round(alpha * 255))

        return r, g, b, a


FOREGROUND_COLOR = (204, 204, 204)
BACKGROUND_COLOR = (31, 31, 31)


def invert_figure(fig: Figure, foreground=FOREGROUND_COLOR, background=BACKGROUND_COLOR) -> Image.Image:
    tmp_path = Path().home() / f'.cache/tmp.{urandom(8).hex()}.{time_ns()}.png'
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(tmp_path)
        img = Image.open(tmp_path)
        inverter = ImageInverter(foreground=foreground, background=background)
        if inverter.should_invert(img):
            img = inverter.invert_image(img)
        return img
    finally:
        tmp_path.unlink(missing_ok=True)

type FileType =  str | bytes | PathLike[str] | PathLike[bytes] | IO[bytes]

# noinspection PyShadowingBuiltins
def save(fig: Figure, fp: FileType, close: bool = True, format: str | None = None, **params: Any) -> None:
    invert_figure(fig).save(fp, format, **params)
    if close:
        plt.close()