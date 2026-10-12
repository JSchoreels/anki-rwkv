"""Validate preparation, recurrence tiles, casting and CUDA Graph experiments."""
from dataclasses import asdict
from pathlib import Path
import gc, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
from rwkv import data_processing as processing
from rwkv.prepare_batch import prepare
from rwkv.model import nvrtc
from scripts.gpu_optimizations import prepare_fast, GraphTrainingAdapter, copy_downcast_fast, transfer_grads_fast
from scripts.check_gpu_perf import assert_equal
from scripts.check_cuda import check_recurrence, synthetic_batch
from scripts.benchmark_gpu_trial import benchmark_models
from scripts.gpu_packed_encodings import prepare_gpu_packed
from rwkv.train_rwkv import transfer_child_grad_to_master
from rwkv.config import DAY_OFFSET_ENCODE_PERIODS
from scripts.experiment_provenance import process_provenance


def check_preparation():
    records = []
    for user in (1,17,92):
        frame=processing.get_rwkv_data(ROOT/'data/anki-revlogs-10k', user)
        for offset in (0, 4096, max(0,len(frame)-190)):
            selected=frame.iloc[offset:offset+190]
            samples=[processing.create_sample(user,selected.iloc[start:start+128],[],torch.float32,torch.device('cpu'))
                     for start in range(0,len(selected),128)]
            original=prepare(samples,seed=2025+offset)
            fast=prepare_fast(samples,seed=2025+offset)
            assert_equal(original, fast)
            deferred=prepare_fast(samples,seed=2025+offset,gpu_encodings=True).to('cuda')
            torch.testing.assert_close(deferred.start.cpu(),original.start,atol=1e-6,rtol=1e-5)
            original_bf16=original.start.to(torch.bfloat16)
            deferred_bf16=deferred.start.to(torch.bfloat16).cpu()
            deferred_different=int((deferred_bf16 != original_bf16).sum().item())
            # Exact indexes and labels; only GPU trigonometry may round differently.
            deferred.start=original.start
            assert_equal(original, deferred.to('cpu'))
            packed=prepare_gpu_packed(samples,seed=2025+offset).to('cuda')
            torch.testing.assert_close(packed.start.cpu(),original.start,atol=1e-6,rtol=1e-5)
            packed_bf16=packed.start.to(torch.bfloat16).cpu()
            packed_different=int((packed_bf16 != original_bf16).sum().item())
            packed.start=original.start
            assert_equal(original,packed.to('cpu'))
            records.append({'user':user,'offset':offset,'reviews':len(selected),
                            'bf16_feature_values':original_bf16.numel(),
                            'gpu_bf16_different_values':deferred_different,
                            'gpu_packed_bf16_different_values':packed_different})
    print('Preparation parity passed: CPU exact; GPU features within 1e-6 absolute / 1e-5 relative',flush=True)
    return records


def check_calendar_precision():
    """Cover every integer calendar remainder before the training BF16 cast."""
    records = []
    for period in DAY_OFFSET_ENCODE_PERIODS:
        remainders = torch.arange(period, dtype=torch.int64)
        angles = (2 * np.pi / period) * remainders
        reference = torch.stack((angles.sin(), angles.cos()), -1).to(torch.bfloat16)
        gpu_remainders = remainders.cuda()
        packed_angles = torch.tensor(2 * np.pi / period, dtype=torch.float32,
                                     device="cuda") * gpu_remainders
        unpacked_angles = (2 * np.pi / torch.tensor(period, device="cuda")) * gpu_remainders
        packed = torch.stack((packed_angles.sin(), packed_angles.cos()), -1).to(torch.bfloat16).cpu()
        unpacked = torch.stack((unpacked_angles.sin(), unpacked_angles.cos()), -1).to(torch.bfloat16).cpu()
        torch.testing.assert_close(packed, reference, rtol=0, atol=0)
        records.append({"period": period, "values": reference.numel(),
                        "packed_bf16_different": int((packed != reference).sum()),
                        "unpacked_bf16_different": int((unpacked != reference).sum())})
    result = {"periods": records, "values": sum(item["values"] for item in records),
              "packed_bf16_different": sum(item["packed_bf16_different"] for item in records),
              "unpacked_bf16_different": sum(item["unpacked_bf16_different"] for item in records)}
    print("Packed calendar features match CPU exactly after BF16 casting for all integer remainders", flush=True)
    return result


def check_tiles():
    records=[]
    nvrtc.register()
    for tile in (32,16,8):
        nvrtc.set_checkpoint_length(tile)
        for dtype in (torch.float32,torch.bfloat16):
            for heads in (1,4):
                check_recurrence(torch.device('cuda:0'), dtype, heads, length=256)
            kernels=nvrtc.module(torch.device('cuda:0'),dtype)
            records.append({'tile':tile,'dtype':str(dtype),'forward':kernels[0].attributes,'backward':kernels[1].attributes})
        print(f'Tile {tile}: float32/BF16 forward and six input gradients match the CPU reference',flush=True)
    nvrtc.set_checkpoint_length(32)
    return records


def check_copies_and_graphs(graph=False, model_names=("baseline", "expertium")):
    records=[]
    for model_name in model_names:
        config={'MODEL':model_name,'DTYPE':'bfloat16','WKV_BACKEND':'nvrtc','SEED':2025}
        torch.manual_seed(2025)
        master,model,dtype=benchmark_models(config,torch.device('cuda:0'))
        with torch.no_grad():
            # Exercise gradients through all initialized-zero projection paths.
            for parameter in master.parameters():
                parameter.add_(torch.randn_like(parameter)*0.002)
        model.copy_downcast_(master,dtype)
        before={name:param.clone() for name,param in model.named_parameters()}
        copy_downcast_fast(master,model)
        for name,param in model.named_parameters():
            torch.testing.assert_close(param,before[name],rtol=0,atol=0)
        del before
        batch=synthetic_batch(torch.device('cuda:0'),dtype)
        # Disable dropout for a deterministic comparison of outputs and gradients.
        model.eval()
        stats=model.get_loss(batch); stats.average_loss.backward()
        child_grads={name:param.grad.clone() for name,param in model.named_parameters() if param.grad is not None}
        transfer_child_grad_to_master(master,model)
        original={name:param.grad.clone() for name,param in master.named_parameters() if param.grad is not None}
        master.zero_grad(set_to_none=True)
        for name,param in model.named_parameters():
            if name in child_grads:
                param.grad.copy_(child_grads[name])
        transfer_grads_fast(master,model)
        for name,param in master.named_parameters():
            if name in original:
                torch.testing.assert_close(param.grad,original[name],rtol=0,atol=0)
        reference_loss=stats.average_loss.item()
        if graph:
            model.zero_grad(set_to_none=True); master.zero_grad(set_to_none=True)
            adapter=GraphTrainingAdapter(model)
            actual=adapter.graph_update(batch)
            torch.testing.assert_close(actual.average_loss,stats.average_loss,atol=2e-6,rtol=1e-5)
            transfer_grads_fast(master,adapter)
            for name,param in master.named_parameters():
                if name in original:
                    torch.testing.assert_close(param.grad,original[name],atol=2e-4,rtol=0.05)
            # Exercise a second graph replay, after resetting gradients.
            model.zero_grad(set_to_none=True)
            actual=adapter.graph_update(batch)
            assert torch.isfinite(actual.average_loss)
            del adapter,actual
        records.append({'model':model_name,'loss':reference_loss,'copies_exact':True,'graph':graph})
        del stats,original,child_grads,master,model,batch
        gc.collect();torch.cuda.empty_cache()
        print(f'{model_name}: casting/gradient parity'+(' and CUDA Graph output/gradient parity' if graph else '')+' passed',flush=True)
    return records


if __name__=='__main__':
    provenance=process_provenance()
    torch.set_num_threads(4)
    nvrtc.register()
    output=ROOT/'artifacts/benchmarks/perf-optimizations'; output.mkdir(parents=True,exist_ok=True)
    if '--graphs-only' in sys.argv:
        nvrtc.set_raw_pointer_launch(True)
        check_recurrence(torch.device('cuda:0'),torch.float32,4,length=256)
        check_recurrence(torch.device('cuda:0'),torch.bfloat16,4,length=256)
        report={'graphs':check_copies_and_graphs(graph=True)}
        name='graph-verification.json'
    else:
        report={'preparation':check_preparation(),'tiles':check_tiles(),'copies':check_copies_and_graphs()}
        name='verification.json'
    report['provenance']=provenance
    (output/name).write_text(json.dumps(report,indent=2)+'\n')
    print('All selected optimization checks passed',flush=True)
