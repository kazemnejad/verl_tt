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

import hashlib
import inspect
import pickle
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def compute_cache_key(cls: type, config: dict | DictConfig) -> str:
    """Compute a deterministic cache key from a class's source file and config.

    The key has the format ``{impl_hash}_{config_hash}`` where both components
    are 12-character lowercase hex strings derived from SHA-256 digests.

    Args:
        cls: The class whose source file is hashed.
        config: The configuration dict or OmegaConf DictConfig to hash.

    Returns:
        A string of the form ``"<impl_hash>_<config_hash>"``.
    """
    # Hash the source file bytes of the class
    impl_hash = hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()[:12]

    # Normalize OmegaConf to plain container before pickling
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()[:12]

    return f"{impl_hash}_{config_hash}"
