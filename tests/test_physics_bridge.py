"""Original fixtures. Native tests require explicit local SDK/model paths.

L2D_PHYSICS_BRIDGE=/private/liblive2d_physics_bridge.so
L2D_TEST_MOC=/private/authorized.moc3
L2D_TEST_PHYSICS=/private/authorized.physics3.json  (optional)
L2D_TEST_POSE_PARTS=VerifiedHairA,VerifiedHairB,VerifiedHairLink (optional)

No model data, native libraries, or official fixtures are embedded here.
"""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from live2d_blender.data import Live2DError
from live2d_blender.physics import OfficialPhysics, _json_blob, validate_physics, validate_pose


def physics_fixture(input_type='Angle', output_type='Angle', reflect=False, weight=100, particles=3):
    """A new, three-particle pendulum using ordinary head-angle controls."""
    group = {
        'Id': 'OriginalTestPendulum',
        'Input': [{'Source': {'Target': 'Parameter', 'Id': 'ParamAngleX'},
                   'Weight': 100, 'Type': input_type, 'Reflect': True}],
        'Output': [{'Destination': {'Target': 'Parameter', 'Id': 'ParamAngleZ'},
                    'VertexIndex': particles - 1, 'Scale': 10, 'Weight': weight,
                    'Type': output_type, 'Reflect': reflect}],
        'Vertices': [{'Position': {'X': 0, 'Y': i * 10}, 'Mobility': .8, 'Delay': .8,
                      'Acceleration': 1, 'Radius': 10 if i else 0} for i in range(particles)],
        'Normalization': {key: {'Minimum': -10, 'Default': 0, 'Maximum': 10}
                          for key in ('Position', 'Angle')},
    }
    return {'Version': 3, 'Meta': {
        'PhysicsSettingCount': 1, 'TotalInputCount': 1, 'TotalOutputCount': 1,
        'VertexCount': particles, 'Fps': 60,
        'EffectiveForces': {'Gravity': {'X': 0, 'Y': -1}, 'Wind': {'X': 0, 'Y': 0}},
    }, 'PhysicsSettings': [group]}


class PhysicsSchemaTests(unittest.TestCase):
    def test_original_fixture(self):
        stats = validate_physics(physics_fixture())
        self.assertEqual((stats['PhysicsSettingCount'], stats['VertexCount'], stats['fps']), (1, 3, 60))

    def test_all_axes_accepted(self):
        for axis in ('X', 'Y', 'Angle'): validate_physics(physics_fixture(axis, axis))

    def test_zero_and_reversed_normalization(self):
        data = physics_fixture()
        data['PhysicsSettings'][0]['Normalization']['Position'] = {'Minimum': 0, 'Maximum': 0, 'Default': 0}
        data['PhysicsSettings'][0]['Normalization']['Angle'] = {'Minimum': 10, 'Maximum': -10, 'Default': 0}
        validate_physics(data)

    def test_metadata_count_mismatch(self):
        data = physics_fixture(); data['Meta']['VertexCount'] = 2
        with self.assertRaises(Live2DError): validate_physics(data)

    def test_bad_output_index(self):
        for index in (-1, 0, 3, 100000, True):
            data = physics_fixture(); data['PhysicsSettings'][0]['Output'][0]['VertexIndex'] = index
            with self.subTest(index=index), self.assertRaises(Live2DError): validate_physics(data)

    def test_bad_type_weight_and_radius(self):
        for key, value in (('Type', 'Unknown'), ('Weight', math.nan), ('Weight', 101), ('Reflect', 'true')):
            data = physics_fixture(); data['PhysicsSettings'][0]['Output'][0][key] = value
            with self.subTest(key=key), self.assertRaises(Live2DError): validate_physics(data)
        data = physics_fixture(); data['PhysicsSettings'][0]['Vertices'][1]['Radius'] = 0
        with self.assertRaises(Live2DError): validate_physics(data)

    def test_empty_chain_rejected(self):
        data = physics_fixture(); data['PhysicsSettings'][0]['Vertices'] = []
        with self.assertRaises(Live2DError): validate_physics(data)

    def test_nonfinite_json_and_nul_id(self):
        with self.assertRaises(Live2DError): _json_blob(b'{"Version":NaN}', 'test')
        data = physics_fixture(); data['PhysicsSettings'][0]['Input'][0]['Source']['Id'] = 'bad\0id'
        with self.assertRaises(Live2DError): validate_physics(data)

    def test_sdk_json_numbers_and_unicode(self):
        data = {'small': 1e-8, 'large': 1e8, '中文': ['中文', 1.25]}
        decoded, raw = _json_blob(data, 'test')
        self.assertEqual(decoded, json.loads(raw))
        self.assertIn(b'0.00000001', raw)
        self.assertIn(b'1.25\n]', raw)
        self.assertNotIn(b'\\u', raw)

    def test_pose_fixture_and_bad_link(self):
        data = {'Type': 'Live2D Pose', 'FadeInTime': .5,
                'Groups': [[{'Id': 'HairA', 'Link': ['HairLink']}, {'Id': 'HairB'}]]}
        self.assertEqual(validate_pose(data)['part_ids'], {'HairA', 'HairB', 'HairLink'})
        data['Groups'][0][0]['Link'] = ['']
        with self.assertRaises(Live2DError): validate_pose(data)

    def test_bad_library_and_json_fail_before_native(self):
        with self.assertRaises(Live2DError): OfficialPhysics('/not/a/bridge.so', b'MOC3', physics_fixture())
        with self.assertRaises(Live2DError): OfficialPhysics('/not/a/bridge.so', b'MOC3', b'{')


BRIDGE = os.environ.get('L2D_PHYSICS_BRIDGE')
MOC = os.environ.get('L2D_TEST_MOC')


@unittest.skipUnless(BRIDGE and MOC, 'Native tests need explicitly selected trusted bridge + authorized MOC')
class OfficialBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moc = Path(MOC).read_bytes()
        with OfficialPhysics(BRIDGE, cls.moc) as rig:
            cls.ids = set(rig.parameter_ids)
        if not {'ParamAngleX', 'ParamAngleZ'} <= cls.ids:
            raise unittest.SkipTest('Original pendulum test requires standard ParamAngleX/ParamAngleZ')

    def new(self, physics=None, pose=None):
        return OfficialPhysics(BRIDGE, self.moc, physics if physics is not None else physics_fixture(), pose)

    def run_signal(self, rig, hz=60, seconds=2):
        rig.reset({'ParamAngleX': 0}, stabilize=True)
        result = []
        for i in range(1, round(hz * seconds) + 1):
            values, parts = rig.evaluate({'ParamAngleX': 20 * math.sin(i / hz * 4)}, {}, 1 / hz)
            self.assertTrue(all(math.isfinite(v) for v in values.values()))
            self.assertTrue(all(math.isfinite(v) for v in parts.values()))
            result.append(values['ParamAngleZ'])
        return result

    def test_reset_and_replay_exact_at_24_30_60(self):
        with self.new() as rig:
            for hz in (24, 30, 60):
                with self.subTest(hz=hz): self.assertEqual(self.run_signal(rig, hz), self.run_signal(rig, hz))

    def test_new_instance_matches_reset_instance(self):
        with self.new() as a, self.new() as b:
            self.run_signal(a, 60, .5)
            self.assertEqual(self.run_signal(a), self.run_signal(b))

    def test_multi_model_isolation_and_close(self):
        with self.new() as a, self.new() as b:
            expected = self.run_signal(a)
            a.reset(); b.reset()
            actual = []
            for i in range(1, 121):
                v, _ = a.evaluate({'ParamAngleX': 20 * math.sin(i / 60 * 4)}, {}, 1 / 60)
                b.evaluate({'ParamAngleX': -29}, {}, 1 / 30)
                actual.append(v['ParamAngleZ'])
            self.assertEqual(expected, actual)
            b.close(); b.close()
            a.evaluate({}, {}, 1 / 60)
            with self.assertRaises(Live2DError): b.evaluate({}, {}, 1 / 60)

    def test_reflection_and_weight(self):
        with self.new(physics_fixture(reflect=False)) as a, self.new(physics_fixture(reflect=True)) as b:
            va, vb = self.run_signal(a), self.run_signal(b)
            self.assertGreater(max(abs(v) for v in va), .01)
            for x, y in zip(va, vb): self.assertAlmostEqual(x, -y, places=6)
        with self.new(physics_fixture(weight=0)) as rig:
            self.assertTrue(all(v == 0 for v in self.run_signal(rig)))
        with self.new(physics_fixture(weight=50)) as half, self.new(physics_fixture(weight=100)) as full:
            for x, y in zip(self.run_signal(half), self.run_signal(full)): self.assertAlmostEqual(x * 2, y, places=6)

    def test_multisegment_changes_response(self):
        with self.new(physics_fixture(particles=2)) as a, self.new(physics_fixture(particles=4)) as b:
            self.assertNotEqual(self.run_signal(a), self.run_signal(b))

    def test_xyz_inputs_and_outputs_finite(self):
        for axis in ('X', 'Y', 'Angle'):
            with self.subTest(axis=axis), self.new(physics_fixture(axis, axis)) as rig:
                self.run_signal(rig, seconds=.5)

    def test_group_output_propagates_to_later_group(self):
        if 'ParamAngleY' not in self.ids: self.skipTest('ParamAngleY is absent')
        data = physics_fixture()
        second = copy.deepcopy(data['PhysicsSettings'][0])
        second['Input'][0]['Source']['Id'] = 'ParamAngleZ'
        second['Output'][0]['Destination']['Id'] = 'ParamAngleY'
        # Amplify the first chain enough to pass the SDK's movement threshold
        # in the second chain; an otherwise valid tiny signal is suppressed.
        data['PhysicsSettings'][0]['Output'][0]['Scale'] = 1000
        data['PhysicsSettings'].append(second)
        for key in ('PhysicsSettingCount', 'TotalInputCount', 'TotalOutputCount', 'VertexCount'): data['Meta'][key] *= 2
        with self.new(data) as rig:
            rig.reset()
            ys = []
            for i in range(120):
                values, _ = rig.evaluate({'ParamAngleX': 20 * math.sin(i / 20)}, {}, 1 / 60)
                ys.append(values['ParamAngleY'])
            self.assertGreater(max(abs(y) for y in ys), 1e-4)

    def test_bad_time_and_missing_reference(self):
        with self.new() as rig:
            for dt in (-1, math.nan, math.inf, 6):
                with self.subTest(dt=dt), self.assertRaises(Live2DError): rig.evaluate({}, {}, dt)
        data = physics_fixture(); data['PhysicsSettings'][0]['Input'][0]['Source']['Id'] = 'NotARealModelParameter_Test'
        with self.assertRaises(Live2DError): self.new(data)

    def test_pose_fade_link_and_reset(self):
        parts = os.environ.get('L2D_TEST_POSE_PARTS', '').split(',')
        if len(parts) != 3 or not all(parts): self.skipTest('Need 3 explicitly verified non-watermark part IDs')
        a, b, link = parts
        pose = {'Type': 'Live2D Pose', 'FadeInTime': .5,
                'Groups': [[{'Id': a, 'Link': [link]}, {'Id': b}]]}
        with self.new(pose=pose) as rig:
            def run():
                rig.reset()
                track = []
                for i in range(30):
                    _, opacity = rig.evaluate({a: 0, b: 1}, {}, 1 / 60)
                    self.assertEqual(opacity[a], opacity[link])
                    track.append(opacity[b])
                return track
            first = run(); self.assertEqual(first, run())
            self.assertLess(first[0], first[-1]); self.assertAlmostEqual(first[-1], 1, places=5)

    def test_private_physics_only_when_explicitly_selected(self):
        path = os.environ.get('L2D_TEST_PHYSICS')
        if not path: self.skipTest('No private physics file selected')
        with self.new(Path(path).read_bytes()) as rig:
            self.assertEqual(self.run_signal(rig), self.run_signal(rig))


if __name__ == '__main__': unittest.main(verbosity=2)
