"""Synthetic independent tests; contains no user-model assets."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
import unittest
import numpy as np
from live2d_blender.bind_selection import select_training_rest, select_training_bind


class BindSelectionTests(unittest.TestCase):
    def test_only_recoverable_degenerate_default_changes(self):
        square = np.array([[-1., -1., .1], [1., -1., .1], [1., 1., .1], [-1., 1., .1]])
        point = np.repeat([[5., 6., .2]], 4, axis=0)
        opened = square + [5., 6., .1]
        line = square.copy(); line[:, 1] = 0; line[:, 2] = .3
        base = np.concatenate([square, point, point, line])
        train = np.stack([base.copy() for _ in range(3)])
        train[1, :4, :2] *= 2  # Healthy rest must stay untouched.
        train[1, 4:8] = opened
        train[2, 4:8] = opened + [30., 40., 0.]
        train[2, 12:16, :2] = square[:, :2] * 2
        pieces = [{'id': str(i), 'offset': i * 4, 'count': 4} for i in range(4)]
        result = select_training_rest(train, pieces, sample_indices=[0, 2, 4])
        np.testing.assert_array_equal(result.selected_sample_indices, [0, 2, 0, 4])
        np.testing.assert_array_equal(result.rest[:4], base[:4])
        np.testing.assert_array_equal(result.rest[8:12], base[8:12])
        np.testing.assert_array_equal(result.rest[:, 2], base[:, 2])
        self.assertEqual(result.summary['corrected_to_rank2_pieces'], 2)
        self.assertEqual(result.summary['still_degenerate_pieces'], 1)
        self.assertEqual(result.summary['static_default_degenerate_pieces'], 1)
        json.dumps(result.summary)
        rest, report = select_training_bind(train, pieces)
        np.testing.assert_array_equal(rest, result.rest)
        json.dumps(report)

    def test_covariance_and_area_are_distinct_objectives(self):
        # Wide thin rectangle has more area but worse minor-axis covariance.
        square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        train = np.stack([np.zeros_like(square), square * [10., .5], square])
        pieces = [{'offset': 0, 'count': 4}]
        tri = np.array([[0, 1, 2], [0, 2, 3]])
        covariance = select_training_rest(train, pieces, tri, method='covariance')
        area = select_training_rest(train, pieces, tri, method='area')
        self.assertEqual(covariance.selected_sample_indices[0], 2)
        self.assertEqual(area.selected_sample_indices[0], 1)
        self.assertEqual(covariance.pieces[0]['selected_min_covariance'], 1.)
        self.assertEqual(area.pieces[0]['selected_area'], 20.)

    def test_training_only_input_is_sufficient_and_deterministic(self):
        # Missing/forbidden held-out labels never enter the function.
        square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        train = np.stack([square * 0, square, square * 2])
        first = select_training_rest(train, [{'offset': 0, 'count': 4}], sample_indices=[0, 2, 60])
        second = select_training_rest(train.copy(), [{'offset': 0, 'count': 4}], sample_indices=[0, 2, 60])
        self.assertEqual(first.selected_sample_indices[0], 60)
        np.testing.assert_array_equal(first.rest, second.rest)
        self.assertEqual(first.pieces, second.pieces)

    def test_inputs_are_not_modified(self):
        train = np.arange(36., dtype=np.float32).reshape(3, 4, 3)
        before = train.copy()
        select_training_bind(train, [{'offset': 0, 'count': 4}])
        np.testing.assert_array_equal(train, before)

    def test_rank_precedes_area_and_zero_shapes_are_retained(self):
        train = np.zeros((3, 4, 2))
        result = select_training_rest(train, [{'offset': 0, 'count': 4}],
                                      np.array([[0, 1, 2]]), method='area')
        self.assertEqual(result.summary['changed_rest_pieces'], 0)
        self.assertEqual(result.summary['still_degenerate_pieces'], 1)

    def test_invalid_partition_and_cross_piece_triangles(self):
        train = np.zeros((2, 8, 2))
        with self.assertRaises(ValueError):
            select_training_bind(train, [{'offset': 0, 'count': 4}])
        pieces = [{'offset': 0, 'count': 4}, {'offset': 4, 'count': 4}]
        with self.assertRaises(ValueError):
            select_training_rest(train, pieces, np.array([[0, 1, 4]]))
        with self.assertRaises(ValueError):
            select_training_rest(train, pieces, sample_indices=[0, 0])
        with self.assertRaises(ValueError):
            select_training_rest(train, pieces, default_sample_index=99)
        with self.assertRaises(ValueError):
            select_training_rest(train, pieces, method='area')


if __name__ == '__main__':
    unittest.main()
