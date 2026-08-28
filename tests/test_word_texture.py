"""Prepared target words switch without touching the visible texture."""

from __future__ import annotations

import unittest

from mantra_hands.render.word_texture import WordTexture


class FakeTexture:
    def __init__(self) -> None:
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.writes = 0
        self.mipmap_builds = 0

    def write(self, data: bytes) -> None:
        self.writes += 1

    def build_mipmaps(self) -> None:
        self.mipmap_builds += 1


class FakeContext:
    def __init__(self) -> None:
        self.textures = []

    def texture(self, *args, **kwargs) -> FakeTexture:  # noqa: ANN002, ANN003
        texture = FakeTexture()
        self.textures.append(texture)
        return texture


class WordTextureTests(unittest.TestCase):
    def test_prepare_keeps_current_word_visible_until_activation(self) -> None:
        word_texture = WordTexture(FakeContext())
        original = word_texture.texture

        word_texture.prepare("borrow")

        self.assertEqual(word_texture.word, "READY")
        self.assertIs(word_texture.texture, original)

        word_texture.activate("borrow")

        self.assertEqual(word_texture.word, "borrow")
        self.assertIsNot(word_texture.texture, original)


if __name__ == "__main__":
    unittest.main()
