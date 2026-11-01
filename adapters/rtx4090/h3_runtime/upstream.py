"""Import the pinned external Sana runtime without vendoring its source tree."""
from importlib import import_module, util
from pathlib import Path
import sys


def runtime(sana_root, module):
    root = Path(sana_root).resolve(strict=True) / 'models/minimax_h3/Sol-H3-RTX5090/runtime'
    name = '_sol_h3_upstream_runtime'
    if name not in sys.modules:
        spec = util.spec_from_file_location(name, root / '__init__.py',
                                           submodule_search_locations=[str(root)])
        package = util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    elif Path(sys.modules[name].__file__).parent != root:
        raise RuntimeError('Cannot mix two Sana source snapshots in one process')
    return import_module(name + '.' + module)


def ltx_paths(root):
    root = Path(root).resolve(strict=True)
    for name in ('ltx-core', 'ltx-pipelines', 'ltx-kernels'):
        source = root / 'packages' / name / 'src'
        if not source.is_dir():
            raise FileNotFoundError(source)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
