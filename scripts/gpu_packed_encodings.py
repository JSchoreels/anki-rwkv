"""Pack deferred feature metadata in CPU workers before IPC and GPU transfer."""
from dataclasses import dataclass
import numpy as np
import torch
from rwkv.config import DAY_OFFSET_ENCODE_PERIODS
from rwkv.model.srs_model import PreparedBatch
from scripts.gpu_optimizations import prepare_fast


@dataclass
class PackedPreparedBatch(PreparedBatch):
    packed: dict

    def to(self, device):
        batch=super().to(device)
        packed={name:value.to(device) for name,value in self.packed.items()}
        columns=[batch.start]
        for index in range(4):
            columns.append(packed[f'table{index}'][packed[f'indices{index}']])
        angles=packed['frequencies']*((packed['baselines']+packed['days'][:,None])%packed['periods'])
        first_angles=packed['frequencies']*((packed['baselines']+packed['first'][:,None])%packed['periods'])
        columns.append(torch.stack((angles.sin(),angles.cos(),first_angles.sin(),first_angles.cos()),-1).flatten(1))
        batch.start=torch.cat(columns,-1)
        batch.start.masked_fill_(packed['padded'][:,None],0)
        return batch


def prepare_gpu_packed(samples, seed, *, batched_id_encodings=False, layout=None):
    deferred=prepare_fast(samples,seed=seed,gpu_encodings=True,
                          batched_id_encodings=batched_id_encodings,layout=layout)
    packed={}
    for index in range(4):
        tables,indices,offset=[],[],0
        for parts,days,first,baselines in deferred.encodings:
            table,inverse=parts[index]
            tables.append(table)
            indices.append(torch.nn.functional.pad(inverse+offset,(0,deferred.global_T-len(inverse))))
            offset+=len(table)
        packed[f'table{index}']=torch.cat(tables)
        packed[f'indices{index}']=torch.cat(indices)
    packed['days']=torch.cat([torch.nn.functional.pad(part[1],(0,deferred.global_T-len(part[1]))) for part in deferred.encodings])
    packed['first']=torch.cat([torch.nn.functional.pad(part[2],(0,deferred.global_T-len(part[2]))) for part in deferred.encodings])
    packed['baselines']=torch.stack([part[3] for part in deferred.encodings]).repeat_interleave(deferred.global_T,0)
    packed['periods']=torch.tensor(DAY_OFFSET_ENCODE_PERIODS)
    # Match the original CPU scalar frequency rounding before GPU multiplication.
    packed['frequencies']=torch.tensor([2*np.pi/period for period in DAY_OFFSET_ENCODE_PERIODS],dtype=torch.float32)
    packed['padded']=torch.cat([torch.arange(deferred.global_T)>=len(part[1]) for part in deferred.encodings])
    fields={key:value for key,value in deferred.__dict__.items() if key not in ('encodings','global_T')}
    return PackedPreparedBatch(**fields,packed=packed)
