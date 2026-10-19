"""Offline independent animation converter: research prototype for Blender 4.2+."""
bl_info = {
    'name': 'Live2D Offline Converter (Research Prototype)',
    'author': 'Live2D Blender Research',
    'version': (0, 3, 0),
    'blender': (4, 2, 0),
    'location': 'File > Import; 3D View > Sidebar > Live2D',
    'description': 'Offline Live2D sampling to independent bones, weights and native effect Actions',
    'category': 'Import-Export',
    'warning': 'Approximate fitting; native masks/blends require Cycles and documented limits.',
}

try:
    import bpy
except ImportError:
    bpy = None

if bpy is not None:
    import json
    from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty, PointerProperty, EnumProperty
    from bpy_extras.io_utils import ImportHelper
    from .assets import load_model3
    from .core import NativeBackend
    from .fixture import SyntheticBackend
    from . import blender_bridge as bridge

    class L2DSettings(bpy.types.PropertyGroup):
        physics_enabled: BoolProperty(name='Official Physics/Pose replay', default=False)
        simulation_fps: FloatProperty(name='Replay simulation FPS', default=60, min=1, max=240)
        simulation_max_steps: IntProperty(name='Replay step limit', default=12000, min=1, max=120000)
        motion_path: StringProperty(name='Single motion3 clip', subtype='FILE_PATH')
        expression_path: StringProperty(name='Single exp3 expression', subtype='FILE_PATH')
        expression_weight: FloatProperty(name='Expression weight', default=1, min=0, max=1)
        play_motion: BoolProperty(name='Evaluate single clip', default=True)
        start_frame: IntProperty(name='Clip start frame', default=1)
        depth_step: FloatProperty(name='Layer depth step', default=0.0001, min=0.000001, max=0.1, precision=6)
        bake_start: IntProperty(name='Bake start', default=1)
        bake_end: IntProperty(name='Bake end', default=60)
        bake_step: IntProperty(name='Sample step', default=1, min=1, max=120)
        skin_bones: IntProperty(name='Logical influences (2 bones each)', default=8, min=1, max=128)
        skin_grouping: EnumProperty(name='Fit grouping', items=(
            ('GLOBAL', 'Global motion clustering', 'General default; no anatomy inferred'),
            ('DRAWABLE', 'Separate each Drawable', 'Hard boundaries; may need many bones and may fit worse'),
            ('MANUAL', 'Manual semantic mapping', 'User-authored JSON groups and bone budgets, no automatic anatomy')), default='GLOBAL')
        skin_mapping_text: StringProperty(name='Manual mapping Text')
        skin_repair_bind: BoolProperty(name='Repair degenerate bind from training poses', default=True)
        skin_separate_static: BoolProperty(name='Independent static influence', default=True)
        skin_static_tolerance: FloatProperty(name='Static tolerance (model units)', default=1e-6, min=0, max=0.01, precision=8)
        skin_subdivisions: EnumProperty(name='Matrix key subdivisions', items=tuple((str(n), str(n), 'Generated only from fitted training matrices') for n in (1, 3, 5, 9, 15, 31)), default='9')
        skin_iterations: IntProperty(name='Fit iterations', default=20, min=1, max=100)
        skin_native_effects: BoolProperty(name='Bake native opacity / tint / layer effects', default=True)
        skin_appearance: EnumProperty(name='Appearance', items=(
            ('STRICT', 'Basic / strict', 'Reject masks and non-normal blends unless explicitly dropped below'),
            ('NATIVE_APPROXIMATE', 'Native masks + blends (Cycles approximation)', 'Bake alpha-only sequences and scene-linear legacy blend nodes; no general SDK pixel parity')), default='STRICT')
        skin_allow_opaque_multiply: BoolProperty(name='Accept opaque-backdrop-only multiply', default=False,
            description='Requires Film Transparent off; multiply changes fractional destination alpha')
        skin_mask_resolution: IntProperty(name='Mask resolution per group', default=512, min=32, max=2048)
        skin_mask_directory: StringProperty(name='Mask output directory', subtype='DIR_PATH', default='//live2d_masks')
        skin_allow_lossy: BoolProperty(name='Drop masks / non-normal blends (lossy)', default=False)
        reference_resolution: IntProperty(name='Reference image long edge', default=512, min=64, max=2048)
        reference_texture_limit: IntProperty(name='Reference texture limit', default=2048, min=256, max=8192)
        show_parts: BoolProperty(name='Show part opacities', default=False)
        filter: StringProperty(name='Parameter filter')

    class L2DPreferences(bpy.types.AddonPreferences):
        bl_idname = __package__
        core_path: StringProperty(name='Official Cubism Core shared library', subtype='FILE_PATH')
        framework_path: StringProperty(name='Locally built official Framework bridge (optional)', subtype='FILE_PATH')

        def draw(self, context):
            layout = self.layout
            layout.label(text='Choose a trusted SDK library that you installed and licensed yourself.')
            layout.prop(self, 'core_path')
            layout.prop(self, 'framework_path')
            layout.label(text='Loading a library executes native code. No Core is bundled or downloaded.')
            layout.label(text='The synthetic demo works without any SDK.')
            layout.label(text='Tested: Linux x86_64, Blender 4.3.2. Windows/macOS native bridges unverified.')

    def core_path(context):
        addon = context.preferences.addons.get(__package__)
        path = addon.preferences.core_path if addon else ''
        if not path:
            raise ValueError('Set the official Cubism Core shared library in Add-on Preferences first')
        return bpy.path.abspath(path)

    class L2D_OT_import(bpy.types.Operator, ImportHelper):
        bl_idname = 'import_scene.live2d_model3'
        bl_label = 'Import Live2D model3 (Experimental)'
        bl_options = {'REGISTER', 'UNDO'}
        filename_ext = '.model3.json'
        filter_glob: StringProperty(default='*.model3.json', options={'HIDDEN'})
        allow_lossy: BoolProperty(name='Allow explicitly lossy preview (NO masks / non-normal blends)', default=False)
        reference_image: BoolProperty(name='Render static reference RGBA instead (offline; hide mesh preview)', default=False)

        def execute(self, context):
            try:
                package = load_model3(self.filepath)
                backend = NativeBackend(package, core_path(context))
                root = bridge.create_model(context, backend, str(package.path), self.allow_lossy or self.reference_image)
                if self.reference_image:
                    from .reference_preview import reference_frame
                    reference_frame(context, root)
                    for obj in bridge.RUNTIMES[root['_l2d_uuid']].objects:
                        obj.hide_set(True)
                        obj.hide_render = True
                warnings = json.loads(root['_l2d_warnings'])
                self.report({'WARNING'}, '; '.join(warnings)[:500] or 'Experimental Core import complete')
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_demo(bpy.types.Operator):
        bl_idname = 'live2d.create_synthetic_demo'
        bl_label = 'Create SDK-free synthetic demo'
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            try:
                bridge.create_model(context, SyntheticBackend())
                self.report({'INFO'}, 'Original synthetic fixture created; this is not a real Live2D import')
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_reconnect(bpy.types.Operator):
        bl_idname = 'live2d.reconnect'
        bl_label = 'Reconnect runtime (explicitly)'
        bl_description = 'Reopen this model using the library in your local preferences; never an embedded path'

        def execute(self, context):
            root = bridge.model_root(context.object)
            if not root:
                return {'CANCELLED'}
            try:
                path = core_path(context) if root['_l2d_kind'] == 'NATIVE' else ''
                bridge.reconnect(context, root, path)
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_refresh(bpy.types.Operator):
        bl_idname = 'live2d.refresh'
        bl_label = 'Update now'

        def execute(self, context):
            root = bridge.model_root(context.object)
            runtime = bridge.runtime_for_root(root)
            if not runtime:
                self.report({'ERROR'}, 'Reconnect runtime first')
                return {'CANCELLED'}
            try:
                root['_l2d_state'] = 'LIVE'
                runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_keyframe(bpy.types.Operator):
        bl_idname = 'live2d.keyframe_parameters'
        bl_label = 'Keyframe all parameters'
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            root = bridge.model_root(context.object)
            if not root:
                return {'CANCELLED'}
            for i, _ in enumerate(json.loads(root.get('_l2d_parameters', '[]'))):
                root.keyframe_insert(data_path=bridge.id_path(bridge.parameter_key(i)), group='Live2D Parameters')
            return {'FINISHED'}

    class L2D_OT_reset(bpy.types.Operator):
        bl_idname = 'live2d.reset_parameters'
        bl_label = 'Reset parameter values'
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            root = bridge.model_root(context.object)
            if not root:
                return {'CANCELLED'}
            for i, parameter in enumerate(json.loads(root.get('_l2d_parameters', '[]'))):
                root[bridge.parameter_key(i)] = float(parameter['default'])
            bridge.update_models(context.scene)
            return {'FINISHED'}

    class L2D_OT_bake(bpy.types.Operator):
        bl_idname = 'live2d.bake_copy'
        bl_label = 'Bake sampled copy (hide source)'
        bl_description = 'Creates sampled mesh shape keys, order and material tracks; not a full Cubism rig conversion'
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            root = bridge.model_root(context.object)
            if not root:
                return {'CANCELLED'}
            try:
                settings = root.live2d
                result = bridge.bake_copy(context, root, settings.bake_start, settings.bake_end, settings.bake_step)
                context.view_layer.objects.active = result
                result.select_set(True)
                self.report({'INFO'}, 'Baked sampled timeline copy; original meshes retained and hidden')
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_independent(bpy.types.Operator):
        bl_idname = 'live2d.convert_independent'
        bl_label = 'Convert to independent bones + native effects'
        bl_description = 'Offline approximate fit; output uses normal Armature weights and Actions without Core or plugin'
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            root = bridge.model_root(context.object)
            if not root:
                return {'CANCELLED'}
            try:
                from .independent_convert import convert_timeline
                settings = root.live2d
                mapping = None
                if settings.skin_grouping == 'MANUAL':
                    text = bpy.data.texts.get(settings.skin_mapping_text)
                    if text is None:
                        raise ValueError('Choose a manual mapping Text, or create a template first')
                    body = text.as_string()
                    if len(body) > 4_000_000:
                        raise ValueError('Manual mapping exceeds 4 MB')
                    mapping = json.loads(body)
                result = convert_timeline(context, root, settings.bake_start, settings.bake_end,
                                          settings.bake_step, settings.skin_bones, settings.skin_iterations,
                                          settings.skin_native_effects, settings.skin_allow_lossy,
                                          grouping=settings.skin_grouping, manual_mapping=mapping,
                                          repair_bind=settings.skin_repair_bind, separate_static=settings.skin_separate_static,
                                          static_tolerance=settings.skin_static_tolerance,
                                          subdivisions=int(settings.skin_subdivisions),
                                          appearance_mode=settings.skin_appearance,
                                          allow_opaque_multiply=settings.skin_allow_opaque_multiply,
                                          mask_resolution=settings.skin_mask_resolution,
                                          mask_directory=settings.skin_mask_directory)
                bpy.ops.object.select_all(action='DESELECT')
                result['armature'].select_set(True)
                context.view_layer.objects.active = result['armature']
                self.report({'INFO'}, f"Independent animation created: {result['report']['actual_bones']} bones. Check conversion_report for fitting error and appearance losses.")
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_mapping_template(bpy.types.Operator):
        bl_idname = 'live2d.create_grouping_template'
        bl_label = 'Create editable manual grouping JSON'
        bl_description = 'Creates a new Text with actual Drawable IDs, not automatic anatomy; edit groups and bone budgets'
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            root = bridge.model_root(context.object)
            runtime = bridge.runtime_for_root(root)
            if runtime is None:
                self.report({'ERROR'}, 'Reconnect the conversion source first')
                return {'CANCELLED'}
            from .fit_pipeline import grouping_template
            pieces = [{'id': d.id} for d in runtime.last_snapshot.drawables]
            text = bpy.data.texts.new('Live2D_manual_groups.json')
            text.write(json.dumps(grouping_template(pieces, root.live2d.skin_bones), indent=2, ensure_ascii=False))
            root.live2d.skin_mapping_text = text.name
            self.report({'INFO'}, 'Open the new Text in Blender Text Editor; edit group membership and budgets before conversion')
            return {'FINISHED'}

    class L2D_OT_connect_physics(bpy.types.Operator):
        bl_idname = 'live2d.connect_physics'
        bl_label = 'Connect official Physics/Pose bridge'
        bl_description = 'Explicitly load your locally built SDK Framework shim; no bundled solver'

        def execute(self, context):
            root = bridge.model_root(context.object)
            runtime = bridge.runtime_for_root(root)
            if not runtime:
                self.report({'ERROR'}, 'Reconnect model runtime first')
                return {'CANCELLED'}
            try:
                addon = context.preferences.addons.get(__package__)
                path = addon.preferences.framework_path if addon else ''
                if not path:
                    raise ValueError('Set locally built Framework bridge in Add-on Preferences first')
                runtime.connect_physics(bpy.path.abspath(path))
                runtime.evaluate(context.scene, context.evaluated_depsgraph_get(), force=True)
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_OT_reference(bpy.types.Operator):
        bl_idname = 'live2d.render_reference'
        bl_label = 'Render current frame to reference RGBA'
        bl_description = 'Offline CPU image with clipping and old blend modes; not realtime or GPU parity certified'

        def execute(self, context):
            root = bridge.model_root(context.object)
            if not root:
                return {'CANCELLED'}
            try:
                from .reference_preview import reference_frame
                image = reference_frame(context, root, root.live2d.reference_resolution, root.live2d.reference_texture_limit)
                self.report({'INFO'}, f'Reference image created: {image.name}. Select it in an Image Editor.')
                return {'FINISHED'}
            except Exception as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}

    class L2D_PT_panel(bpy.types.Panel):
        bl_label = 'Live2D Offline Converter'
        bl_idname = 'L2D_PT_runtime_bridge'
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = 'Live2D'

        def draw(self, context):
            layout = self.layout
            layout.label(text='Research prototype 0.2 • not full parity', icon='ERROR')
            layout.operator('import_scene.live2d_model3', text='Import model3.json')
            layout.operator('live2d.create_synthetic_demo')
            root = bridge.model_root(context.object)
            if not root:
                if context.object and context.object.get('conversion_report'):
                    report = json.loads(context.object['conversion_report'])
                    layout.label(text='Independent Blender animation', icon='ARMATURE_DATA')
                    layout.label(text='Playback needs no Core or add-on')
                    layout.label(text=f"{report['actual_bones']} bones; training RMS {report['training_metrics']['rms']:.6g}")
                    layout.label(text='Training fit is not a full visual certification')
                return
            layout.separator()
            layout.label(text=root.name)
            state = root.get('_l2d_state') if bridge.runtime_for_root(root) or root.get('_l2d_kind') == 'BAKED' else 'DISCONNECTED'
            layout.label(text=f"Backend: {root.get('_l2d_kind')} | {state}")
            if root.get('_l2d_kind') == 'BAKED':
                layout.label(text='Sampled timeline; no Core required for playback')
                return
            if root.get('_l2d_runtime_error'):
                layout.label(text=root['_l2d_runtime_error'][:80], icon='ERROR')
            layout.operator('live2d.reconnect')
            layout.operator('live2d.refresh')
            if root.get('_l2d_kind') == 'NATIVE':
                layout.operator('live2d.connect_physics')
                layout.prop(root.live2d, 'physics_enabled')
                if root.live2d.physics_enabled:
                    layout.prop(root.live2d, 'simulation_fps')
            box = layout.box()
            box.label(text='Motion / expression: single clip, no fade mixer')
            box.prop(root.live2d, 'motion_path')
            box.prop(root.live2d, 'play_motion')
            box.prop(root.live2d, 'start_frame')
            box.prop(root.live2d, 'expression_path')
            box.prop(root.live2d, 'expression_weight')
            row = layout.row(align=True)
            row.operator('live2d.keyframe_parameters', text='Key all')
            row.operator('live2d.reset_parameters', text='Reset')
            layout.prop(root.live2d, 'filter')
            query = root.live2d.filter.casefold()
            display = json.loads(root.get('_l2d_display_info', '{}'))
            names = {p['Id']: p.get('Name', p['Id']) for p in display.get('Parameters', [])}
            for i, p in enumerate(json.loads(root.get('_l2d_parameters', '[]'))):
                label = names.get(p['id'], p['id'])
                if query and query not in (p['id'] + ' ' + label).casefold():
                    continue
                layout.prop(root, bridge.id_path(bridge.parameter_key(i)), text=label, slider=True)
            layout.prop(root.live2d, 'show_parts')
            if root.live2d.show_parts:
                for i, p in enumerate(json.loads(root.get('_l2d_parts', '[]'))):
                    layout.prop(root, bridge.id_path(bridge.part_key(i)), text=p['id'], slider=True)
            box = layout.box()
            box.label(text='Offline conversion: independent bones + native effects')
            row = box.row(align=True)
            row.prop(root.live2d, 'bake_start')
            row.prop(root.live2d, 'bake_end')
            box.prop(root.live2d, 'bake_step')
            box.prop(root.live2d, 'skin_bones')
            box.prop(root.live2d, 'skin_iterations')
            box.prop(root.live2d, 'skin_grouping')
            if root.live2d.skin_grouping == 'MANUAL':
                box.prop_search(root.live2d, 'skin_mapping_text', bpy.data, 'texts')
                box.operator('live2d.create_grouping_template')
                box.label(text='Edit JSON in Text Editor. No automatic anatomy.')
            if root.live2d.skin_grouping == 'DRAWABLE':
                box.label(text='Each moving Drawable needs its own bone budget.')
            box.prop(root.live2d, 'skin_repair_bind')
            box.prop(root.live2d, 'skin_separate_static')
            if root.live2d.skin_separate_static:
                box.prop(root.live2d, 'skin_static_tolerance')
            box.prop(root.live2d, 'skin_subdivisions')
            box.label(text='Total budget includes an independent static influence.')
            box.prop(root.live2d, 'skin_native_effects')
            box.prop(root.live2d, 'skin_appearance')
            if root.live2d.skin_appearance == 'NATIVE_APPROXIMATE':
                box.prop(root.live2d, 'skin_allow_opaque_multiply')
                box.prop(root.live2d, 'skin_mask_resolution')
                box.prop(root.live2d, 'skin_mask_directory')
                box.label(text='Save .blend first; keep its mask folder with it.')
                box.label(text='Cycles only; check transparent bounce convergence.')
            else:
                box.prop(root.live2d, 'skin_allow_lossy')
            box.operator('live2d.convert_independent')
            box.label(text='Development fallback: sampled shape keys')
            box.operator('live2d.bake_copy')
            box = layout.box()
            box.label(text='Offline reference image: clipping + old blends')
            box.prop(root.live2d, 'reference_resolution')
            box.prop(root.live2d, 'reference_texture_limit')
            box.operator('live2d.render_reference')
            if root.get('_l2d_reference_image'):
                box.label(text=root['_l2d_reference_image'][:70], icon='IMAGE_DATA')
            for warning in json.loads(root.get('_l2d_warnings', '[]'))[:4]:
                layout.label(text=warning[:95], icon='INFO')

    CLASSES = (L2DSettings, L2DPreferences, L2D_OT_import, L2D_OT_demo, L2D_OT_reconnect,
               L2D_OT_refresh, L2D_OT_keyframe, L2D_OT_reset, L2D_OT_bake, L2D_OT_independent, L2D_OT_mapping_template, L2D_OT_connect_physics, L2D_OT_reference, L2D_PT_panel)

    def import_menu(self, context):
        self.layout.operator(L2D_OT_import.bl_idname, text='Live2D Cubism (.model3.json) [Experimental]')

    def register():
        for cls in CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.Object.live2d = PointerProperty(type=L2DSettings)
        bpy.types.TOPBAR_MT_file_import.append(import_menu)
        bridge.register_handlers()

    def unregister():
        bridge.unregister_handlers()
        bpy.types.TOPBAR_MT_file_import.remove(import_menu)
        del bpy.types.Object.live2d
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)
else:
    def register():
        raise RuntimeError('This add-on must be registered inside Blender 4.2+')

    def unregister():
        pass
