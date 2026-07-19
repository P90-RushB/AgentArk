"""Force Transformers to use its PyTorch causal-conv1d fallback.

The locally validated ms-swift 4.4.1 environment combines Torch 2.10 with
causal-conv1d 1.6.2.post1. That native extension segfaults during the Hugging
Face training forward pass. This process-local shim mirrors the known-good
Swift smoke setup without modifying site-packages or vLLM inference.
"""

import transformers.utils.import_utils as import_utils


import_utils.is_causal_conv1d_available = lambda: False
