import unittest
import torch

from sage.native_models import make_native_encoder


class NativeEncoderTests(unittest.TestCase):
    def test_published_shapes(self):
        torch.set_num_threads(2)
        for size, width, heads, image_size in [('tiny', 192, 3, 112), ('small', 384, 6, 224)]:
            model = make_native_encoder({'_target_': 'stable_pretraining.backbone.utils.vit_hf',
                'size': size, 'patch_size': 14, 'image_size': image_size,
                'pretrained': False, 'use_mask_token': False})
            self.assertEqual(model.config.hidden_size, width)
            self.assertEqual(model.config.num_attention_heads, heads)
            self.assertEqual(model.config.num_hidden_layers, 12)
            self.assertEqual(model.config.intermediate_size, 4*width)
            self.assertTrue(model.config.interpolate_pos_encoding)
            self.assertFalse(any('pooler' in key or 'mask_token' in key for key in model.state_dict()))


if __name__ == '__main__':
    unittest.main()
