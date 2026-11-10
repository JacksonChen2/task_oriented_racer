#!/usr/bin/env python3
"""Open every scene dependency using Isaac's USD, without starting Kit/GPU."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('RACER_USD_CHECK_CHILD') != '1':
    isaac = Path(os.environ['ISAAC_SIM_ROOT'])
    matches = sorted((isaac / 'extscache').glob('omni.usd.libs-*/pxr'))
    if not matches:
        raise SystemExit('Cannot locate Isaac USD Python libraries')
    usd = matches[-1].parent
    env = os.environ.copy()
    env['RACER_USD_CHECK_CHILD'] = '1'
    env['PYTHONPATH'] = str(usd)
    env['LD_LIBRARY_PATH'] = str(usd / 'bin') + ':' + env.get('LD_LIBRARY_PATH', '')
    raise SystemExit(subprocess.call([sys.executable, __file__], env=env))

from pxr import Sdf, Usd, UsdUtils

assets = ROOT / 'final_racer/assets'
entry = assets / 'warehouse_scenes/isaac/warehouse_full_with_industrial_ap.usda'
drone = assets / 'isaac_assets/racer_so3_quadrotor/usd/crazyflie_with_racer_dynamics.usd'
pending = [entry, drone]
seen = set()
engine_materials = set()
while pending:
    path = pending.pop().resolve()
    if path in seen:
        continue
    seen.add(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == '.mdl':
        text = path.read_text()
        for module in re.findall(r'(?:using|import)\s+\.::([A-Za-z0-9_]+)', text):
            pending.append(path.parent / (module + '.mdl'))
        for texture in re.findall(r'texture_\w+\(\s*"([^"]+)"', text):
            if texture:
                if '://' in texture or Path(texture).is_absolute():
                    raise ValueError(f'Nonportable MDL texture: {texture}')
                pending.append(path.parent / texture)
        continue
    if path.suffix.lower() not in ('.usd', '.usda', '.usdc'):
        continue
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f'Cannot open USD layer: {path}')
    def inspect(asset):
        if not asset:
            return asset
        if '://' in asset or Path(asset).is_absolute():
            raise ValueError(f'Nonportable USD reference: {path}: {asset}')
        if asset.endswith('.mdl') and '/' not in asset and not (path.parent / asset).is_file():
            candidates = list((Path(os.environ['ISAAC_SIM_ROOT']) / 'kit/mdl').rglob(asset))
            if not candidates:
                raise FileNotFoundError(f'Isaac built-in material missing: {asset}')
            engine_materials.add(asset)
        else:
            dependency = path.parent / asset
            if '<UDIM>' in asset:
                tiles = list(dependency.parent.glob(dependency.name.replace('<UDIM>', '[0-9][0-9][0-9][0-9]')))
                if not tiles:
                    raise FileNotFoundError(f'UDIM tiles missing: {dependency}')
                pending.extend(tiles)
            else:
                pending.append(dependency)
        return asset
    UsdUtils.ModifyAssetPaths(layer, inspect)
stage = Usd.Stage.Open(str(entry), load=Usd.Stage.LoadAll)
if stage is None:
    raise RuntimeError('Cannot compose warehouse')
prims = list(stage.Traverse())
meshes = sum(p.GetTypeName() == 'Mesh' for p in prims)
assert meshes > 0
print(json.dumps({'status': 'SCENE_LOCAL_DEPENDENCIES_OK', 'files_checked': len(seen),
                  'composed_prims': len(prims), 'meshes': meshes,
                  'isaac_builtin_materials': sorted(engine_materials)}))
