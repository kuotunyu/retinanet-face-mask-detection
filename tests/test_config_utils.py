import os
import unittest

from utils.config import PROJECT_ROOT, cfg_get, parse_int_list, resolve_path, str2bool


class ConfigUtilsTest(unittest.TestCase):
    def test_str2bool_accepts_common_values(self):
        for value in [True, 'true', 'True', '1', 'yes', 'y', 'on']:
            self.assertTrue(str2bool(value))
        for value in [False, 'false', 'False', '0', 'no', 'n', 'off']:
            self.assertFalse(str2bool(value))

    def test_str2bool_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            str2bool('maybe')

    def test_parse_int_list(self):
        self.assertEqual(parse_int_list('600,600'), [600, 600])
        self.assertEqual(parse_int_list(' 32, 64 ,128 '), [32, 64, 128])
        self.assertEqual(parse_int_list([600, 600]), [600, 600])
        self.assertEqual(parse_int_list((1, 2)), [1, 2])

    def test_cfg_get_reads_nested_keys(self):
        config = {'model': {'input_shape': [600, 600]}}
        self.assertEqual(cfg_get(config, 'model.input_shape'), [600, 600])
        self.assertIsNone(cfg_get(config, 'model.missing'))
        self.assertEqual(cfg_get(config, 'paths.classes_path', 'fallback'), 'fallback')

    def test_resolve_path(self):
        self.assertIsNone(resolve_path(None))
        self.assertEqual(resolve_path(''), '')
        self.assertEqual(
            resolve_path('configs/mask_retinanet.yaml'),
            os.path.join(PROJECT_ROOT, 'configs/mask_retinanet.yaml')
        )
        absolute = os.path.abspath(os.sep)
        self.assertEqual(resolve_path(absolute), absolute)


if __name__ == '__main__':
    unittest.main()
