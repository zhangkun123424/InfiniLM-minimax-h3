"""Exercise the InfiniCore Hygon TimestepEmbedding dispatch with NineToothed."""
import json
import math
import sys
from pathlib import Path

import infinicore
import torch

rows = []
for index in range(8):
    device = f'cuda:{index}'
    torch.cuda.set_device(index)
    infinicore.set_device(infinicore.device('cuda', index))
    for dim in (16, 256):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for count in (1, 4, 17):
                x = torch.linspace(-1000, 1000, count, device=device, dtype=dtype)
                y = torch.empty((count, dim), device=device, dtype=torch.float32)
                torch.cuda.synchronize(index)
                result = infinicore.timestep_embedding(infinicore.from_torch(x), dim)
                infinicore.sync_device()
                infinicore.from_torch(y).copy_(result)
                infinicore.sync_device()
                torch.cuda.synchronize(index)
                angles = x.float()[:, None] * torch.exp(
                    -math.log(10000) * torch.arange(dim // 2, device=device) / (dim // 2)
                )
                ref = torch.cat((angles.cos(), angles.sin()), dim=1)
                error = (y - ref).abs().max().item()
                if not math.isfinite(error) or error > 0.002:
                    raise RuntimeError(f'{device}, {dim}, {dtype}, {count}: {error}')
                rows.append(dict(device=index, dim=dim, dtype=str(dtype), count=count, max_abs=error))
Path(sys.argv[1]).write_text(json.dumps(rows, indent=2))
