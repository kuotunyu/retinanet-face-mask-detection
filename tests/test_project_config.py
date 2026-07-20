import unittest

from utils.anchors import get_anchors
from utils.config import cfg_get, load_config, resolve_path
from utils.utils import get_classes


class ProjectConfigSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/mask_retinanet.yaml')

    def test_classes_file_matches_config(self):
        classes_path = resolve_path(cfg_get(self.config, 'paths.classes_path'))
        class_names, num_classes = get_classes(classes_path)

        self.assertEqual(num_classes, 3)
        self.assertEqual(
            class_names,
            ['without_mask', 'with_mask', 'mask_weared_incorrect']
        )

    def test_anchor_count_matches_retinanet_layout(self):
        input_shape = cfg_get(self.config, 'model.input_shape')
        anchors_size = cfg_get(self.config, 'model.anchors_size')
        anchors = get_anchors(input_shape, anchors_size)

        self.assertEqual(anchors.shape[1], 4)
        self.assertGreater(anchors.shape[0], 0)
        self.assertTrue((anchors >= 0).all())
        self.assertTrue((anchors <= 1).all())


if __name__ == '__main__':
    unittest.main()
