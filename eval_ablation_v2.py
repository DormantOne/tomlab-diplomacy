from diplomacy_fovea_v2 import enable_fovea_v2
from diplomacy_message_compress import enable_compressed_messages

enable_fovea_v2()
enable_compressed_messages()

import eval_ablation
import sys
sys.exit(eval_ablation.main())