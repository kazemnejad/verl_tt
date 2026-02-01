# Copyright 2025 Individual Contributor: Amirhossein Kazemnejad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import re
import sys
import textwrap

from omegaconf import OmegaConf

from treetune_verl.tasks.cache import compute_cache_key


def _load_class_from_tmp(tmp_path, filename, source, class_name="Dummy"):
    """Helper: write a .py file, import the class from it."""
    filepath = tmp_path / filename
    filepath.write_text(source)
    mod_name = f"_tmp_{filename.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def test_cache_key_deterministic(tmp_path):
    """Same class + same config -> same key."""
    src = textwrap.dedent("""\
        class Dummy:
            pass
    """)
    cls = _load_class_from_tmp(tmp_path, "dummy.py", src)
    config = {"a": 1, "b": "hello"}
    key1 = compute_cache_key(cls, config)
    key2 = compute_cache_key(cls, config)
    assert key1 == key2


def test_cache_key_changes_on_config_change(tmp_path):
    """Different config -> different key."""
    src = textwrap.dedent("""\
        class Dummy:
            pass
    """)
    cls = _load_class_from_tmp(tmp_path, "dummy2.py", src)
    key1 = compute_cache_key(cls, {"a": 1})
    key2 = compute_cache_key(cls, {"a": 2})
    assert key1 != key2


def test_cache_key_changes_on_source_change(tmp_path):
    """Modifying the source file -> different impl_hash."""
    src_v1 = textwrap.dedent("""\
        class Dummy:
            x = 1
    """)
    src_v2 = textwrap.dedent("""\
        class Dummy:
            x = 2
    """)
    cls_v1 = _load_class_from_tmp(tmp_path, "dummy_v1.py", src_v1)
    cls_v2 = _load_class_from_tmp(tmp_path, "dummy_v2.py", src_v2)
    config = {"a": 1}
    key1 = compute_cache_key(cls_v1, config)
    key2 = compute_cache_key(cls_v2, config)
    # Same config but different source files -> different keys
    assert key1 != key2


def test_cache_key_format(tmp_path):
    """Key is '{impl_hash}_{config_hash}' format, both hex strings."""
    src = textwrap.dedent("""\
        class Dummy:
            pass
    """)
    cls = _load_class_from_tmp(tmp_path, "dummy_fmt.py", src)
    key = compute_cache_key(cls, {"x": 42})
    # Format: <hex>_<hex>
    assert re.match(r"^[0-9a-f]+_[0-9a-f]+$", key), f"Unexpected key format: {key}"
    parts = key.split("_")
    assert len(parts) == 2
    assert len(parts[0]) == 12
    assert len(parts[1]) == 12


def test_cache_key_works_with_omegaconf(tmp_path):
    """compute_cache_key accepts OmegaConf DictConfig."""
    src = textwrap.dedent("""\
        class Dummy:
            pass
    """)
    cls = _load_class_from_tmp(tmp_path, "dummy_oc.py", src)
    config_dict = {"a": 1, "b": "hello"}
    config_oc = OmegaConf.create(config_dict)
    key_dict = compute_cache_key(cls, config_dict)
    key_oc = compute_cache_key(cls, config_oc)
    assert key_dict == key_oc
