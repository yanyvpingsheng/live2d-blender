"""Independent small-fixture and rejection-path review, never loads Core."""
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from live2d_blender.fit_pipeline import fit_animation
from live2d_blender.skinning import SkinningFit,predict
from live2d_blender.refine_keys import refine_affine_keys,refined_frame_times
from live2d_blender.bind_selection import select_training_rest


def sample(groups=1,frames=3,static_last=False):
    triangle=np.array([[0.,0.,0.],[.8,0.,0.],[0.,.7,0.]])
    rest=np.concatenate([triangle+[2*i,0,0] for i in range(groups)])
    poses=np.repeat(rest[None],frames,axis=0)
    for t in range(frames):
        for group in range(groups-int(static_last)):
            poses[t,group*3:group*3+3,:2]+=[.1*t*(group+1),-.05*t]
    return SimpleNamespace(positions=poses,triangles=np.asarray([[3*i,3*i+1,3*i+2] for i in range(groups)]),pieces=[{'id':f'piece{i}','offset':3*i,'count':3} for i in range(groups)])


def mapping(*groups):
    return {'version':1,'groups':[{'name':name,'bones':bones,'drawables':ids} for name,bones,ids in groups]}


def fitted():
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    a=np.zeros((2,1,2,3));a[:,:,0,0]=1;a[:,:,1,1]=1
    return SkinningFit(rest,np.ones((3,1)),a,np.zeros((2,1)),np.zeros((1,3)),{})


class PipelineReviewTests(unittest.TestCase):
    def test_all_static_uses_one_bone_in_every_grouping(self):
        s=sample(2,static_last=True);s.positions[:]=s.positions[0]
        for grouping in ('GLOBAL','DRAWABLE','MANUAL'):
            with self.subTest(grouping=grouping):
                f,_,_,r=fit_animation(s,[1,2,3],grouping=grouping,bone_budget=1,manual_mapping=mapping(('unused',128,['piece0','piece1'])))
                self.assertEqual(r['logical_bones'],1);self.assertEqual(r['static_vertices'],6);self.assertEqual(r['groups'],[])
                np.testing.assert_allclose(predict(f),s.positions);json.dumps(r,allow_nan=False)
    def test_all_static_preflight_uses_actual_not_requested_bones(self):
        s=sample(frames=1000);s.positions[:]=s.positions[0]
        _,_,times,r=fit_animation(s,np.arange(1000),bone_budget=128,subdivisions=1)
        self.assertEqual(len(times),1000);self.assertEqual(r['logical_bones'],1)
    def test_one_and_two_moving_points(self):
        for count in (1,2):
            s=sample();s.positions[:]=s.positions[0];s.positions[1:,:count,0]+=.2
            f,_,_,r=fit_animation(s,[1,2,3],bone_budget=2,iterations=1)
            self.assertEqual(r['logical_bones'],2);self.assertEqual(r['static_vertices'],3-count)
            np.testing.assert_allclose(predict(f),s.positions,atol=2e-6)
            # The group metric must use full-model units, not the zero-size
            # repeated solver point's artificial 1e-12 bounding box.
            expected=predict(f)[:, :count];err=np.max(np.linalg.norm(expected-s.positions[:,:count],axis=2))
            reported=max(row['max_at_1024px'] for row in r['groups'][0]['training_metrics']['per_pose'])
            self.assertAlmostEqual(reported,err*1024/.8,places=5)
    def test_mixed_static_needs_reserved_budget(self):
        s=sample();s.positions[:,1:]=s.positions[0,1:]
        with self.assertRaisesRegex(ValueError,'static influence'):fit_animation(s,[1,2,3],bone_budget=1)
    def test_static_separation_can_be_disabled(self):
        s=sample();s.positions[:,1:]=s.positions[0,1:]
        f,_,_,r=fit_animation(s,[1,2,3],bone_budget=1,separate_static=False)
        self.assertEqual(r['static_vertices'],0);self.assertEqual(r['logical_bones'],1)
        np.testing.assert_allclose(predict(f),s.positions,atol=2e-6)
    def test_drawable_groups_do_not_cross_weights(self):
        s=sample(2);f,_,_,r=fit_animation(s,[1,2,3],bone_budget=2,grouping='DRAWABLE')
        self.assertEqual(r['logical_bones'],2)
        np.testing.assert_array_equal(f.weights[:3,1],0);np.testing.assert_array_equal(f.weights[3:,0],0)
    def test_too_many_drawables_rejected_before_fit(self):
        with patch('live2d_blender.fit_pipeline.fit_skinning',side_effect=AssertionError('fit should not run')):
            with self.assertRaisesRegex(ValueError,'moving groups'):fit_animation(sample(3),[1,2,3],bone_budget=2,grouping='DRAWABLE')
    def test_manual_static_omission_and_budgets(self):
        s=sample(3,static_last=True);m=mapping(('left',1,['piece0']),('right',1,['piece1']))
        f,_,_,r=fit_animation(s,[1,2,3],bone_budget=3,grouping='MANUAL',manual_mapping=m)
        self.assertEqual(r['logical_bones'],3);self.assertEqual(r['actual_bones'],6);self.assertEqual(r['unused_budget'],0)
        np.testing.assert_array_equal(f.weights[6:,0],1);np.testing.assert_array_equal(f.weights[:3,2],0)
    def test_static_only_manual_budget_not_allocated(self):
        s=sample(2,static_last=True);m=mapping(('move',1,['piece0']),('static',128,['piece1']))
        _,_,_,r=fit_animation(s,[1,2,3],bone_budget=2,grouping='MANUAL',manual_mapping=m)
        self.assertEqual(r['logical_bones'],2);self.assertEqual(len(r['groups']),1)
    def test_manual_sum_counts_static_reservation(self):
        s=sample(3,static_last=True);m=mapping(('left',1,['piece0']),('right',1,['piece1']))
        with self.assertRaisesRegex(ValueError,'static influence'):fit_animation(s,[1,2,3],bone_budget=2,grouping='MANUAL',manual_mapping=m)
    def test_manual_validation(self):
        bad=[mapping(('a',1,['unknown'])),mapping(('a',1,['piece0','piece0'])),mapping(('a',1,['piece0']),('b',1,['piece0'])),mapping(('a',True,['piece0'])),mapping(('a',1.5,['piece0'])),mapping(('a',1,['piece0']),('a',1,['piece1'])),{'version':True,'groups':[{'name':'a','bones':1,'drawables':['piece0','piece1']}]}]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):fit_animation(sample(2),[1,2,3],grouping='MANUAL',manual_mapping=value)
    def test_manual_missing_moving_drawable(self):
        with self.assertRaisesRegex(ValueError,'Every moving Drawable'):fit_animation(sample(2),[1,2,3],grouping='MANUAL',manual_mapping=mapping(('a',1,['piece0'])))
    def test_manual_capacity(self):
        with self.assertRaisesRegex(ValueError,'too few moving vertices'):fit_animation(sample(),[1,2,3],grouping='MANUAL',manual_mapping=mapping(('a',2,['piece0'])))
    def test_bind_repair_is_training_only_and_preserves_input(self):
        s=sample();s.positions[0]=0;before=s.positions.copy()
        f,_,_,r=fit_animation(s,[10,20,30],bone_budget=2)
        self.assertEqual(r['bind_selection']['summary']['corrected_to_rank2_pieces'],1)
        self.assertIn(r['bind_selection']['selected_training_frames'][0],[20.,30.]);self.assertGreater(np.ptp(f.rest[:,:2]),0)
        np.testing.assert_array_equal(s.positions,before)
    def test_single_training_frame(self):
        s=sample(frames=1);f,_,times,r=fit_animation(s,[3.25],bone_budget=1)
        np.testing.assert_array_equal(times,[3.25]);np.testing.assert_allclose(predict(f),s.positions)
    def test_budget_is_cap_not_forced_unused_bones(self):
        _,_,_,r=fit_animation(sample(),[1,2,3],bone_budget=128)
        self.assertEqual(r['logical_bones'],1);self.assertEqual(r['unused_budget'],127)
    def test_bad_time_options_rejected_before_fit(self):
        for options in ({'subdivisions':2},{'subdivisions':True},{'subdivisions':33}):
            with patch('live2d_blender.fit_pipeline.fit_skinning',side_effect=AssertionError('fit should not run')):
                with self.assertRaises(ValueError):fit_animation(sample(),[1,2,3],**options)
    def test_float32_key_collision_rejected_before_fit(self):
        with patch('live2d_blender.fit_pipeline.fit_skinning',side_effect=AssertionError('fit should not run')):
            with self.assertRaisesRegex(ValueError,'precision'):fit_animation(sample(frames=2),[1000000,1000001],subdivisions=31)
    def test_total_bone_key_limit_before_fit(self):
        s=sample(41,frames=2500)
        with patch('live2d_blender.fit_pipeline.fit_skinning',side_effect=AssertionError('fit should not run')):
            with self.assertRaisesRegex(ValueError,'100000'):fit_animation(s,np.arange(2500),bone_budget=41,subdivisions=1)
    def test_fractional_piece_span_rejected(self):
        s=sample();s.pieces[0]['count']=3.5
        with self.assertRaises(ValueError):fit_animation(s,[1,2,3])


class RefinementReviewTests(unittest.TestCase):
    def test_limits_exact_boundary(self):
        self.assertEqual(len(refined_frame_times(np.arange(20000),1,5)),20000)
        with self.assertRaisesRegex(ValueError,'100000'):refined_frame_times(np.arange(20000),1,6)
        with self.assertRaisesRegex(ValueError,'20000'):refined_frame_times(np.arange(20001),1)
    def test_nonfinite_and_shape_rejected(self):
        for attribute,value in [('affines',np.full((2,1,2,3),np.nan)),('depth_offsets',np.zeros((2,2))),('rest',np.zeros((3,2))),('centers',np.zeros((2,3))),('weights',np.ones((3,1))*2)]:
            f=fitted();setattr(f,attribute,value)
            with self.subTest(attribute=attribute):
                with self.assertRaises(ValueError):refine_affine_keys(f,[1,3])
    def test_invalid_frames_rejected(self):
        for frames in ([],[1],[2,1],[1,1],[1,np.nan],[1,np.inf]):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):refine_affine_keys(fitted(),frames)
    def test_keeps_endpoints_midpoint_not_key(self):
        f=fitted();before=f.affines.copy();out,times=refine_affine_keys(f,[1,3],9)
        np.testing.assert_array_equal(out.affines[[0,-1]],before);np.testing.assert_array_equal(f.affines,before)
        self.assertGreater(abs(times-2).min(),1e-8)
    def test_fifth_weight_rejected(self):
        f=fitted();f.weights=np.full((3,5),.2);f.affines=np.repeat(f.affines,5,axis=1);f.depth_offsets=np.zeros((2,5));f.centers=np.zeros((5,3))
        with self.assertRaisesRegex(ValueError,'four'):refine_affine_keys(f,[1,3])


class BindInputReviewTests(unittest.TestCase):
    def test_nonfinite_tolerances_rejected(self):
        s=sample()
        for args in ({'relative_rank_tolerance':np.nan},{'absolute_std_fraction':np.inf}):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):select_training_rest(s.positions,s.pieces,**args)
    def test_fractional_source_labels_rejected(self):
        s=sample()
        with self.assertRaises(ValueError):select_training_rest(s.positions,s.pieces,sample_indices=[0.,1.25,2.5])
    def test_fractional_piece_count_rejected(self):
        s=sample();s.pieces[0]['count']=3.5
        with self.assertRaises(ValueError):select_training_rest(s.positions,s.pieces)

if __name__=='__main__':unittest.main()
