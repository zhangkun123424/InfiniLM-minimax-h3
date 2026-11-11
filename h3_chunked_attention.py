"""Bound MiniMax-H3 attention workspace by slicing queries, retaining all keys."""
import math
import torch

def install(query_chunk_size=1024, score_mib=2048, min_tokens=2048):
    import diffusers.models.transformers.transformer_minimax_h3 as h3
    original = h3.dispatch_attention_fn
    if getattr(original, '_h3_query_chunked', False):
        return original
    if query_chunk_size < 1 or score_mib < 1:
        raise ValueError('Positive chunk size and score budget are required')
    stats = {'calls': 0, 'chunks': 0}
    def dispatch(query, key, value, *args, **kwargs):
        # H3's layout is batch, sequence, heads, head_dim. Other paths stay native.
        unsafe_grad = torch.is_grad_enabled() and any(x.requires_grad for x in (query,key,value))
        if (args or query.ndim != 4 or key.ndim != 4 or value.ndim != 4
            or kwargs.get('is_causal', False) or kwargs.get('dropout_p', 0.0) != 0.0
            or kwargs.get('parallel_config') is not None or unsafe_grad
            or query.shape[1] <= min_tokens):
            return original(query,key,value,*args,**kwargs)
        length = query.shape[1]
        score_row_bytes = query.shape[0] * query.shape[2] * key.shape[1] * max(4,query.element_size())
        chunk = min(query_chunk_size, max(1, (score_mib * 1024 * 1024) // max(1,score_row_bytes)))
        if chunk >= length:
            return original(query,key,value,**kwargs)
        mask = kwargs.get('attn_mask')
        if mask is not None and mask.ndim == 2:
            if mask.shape == (query.shape[0], key.shape[1]):
                mask = mask[:,None,None,:]
            elif mask.shape[-2] in (1,length):
                mask = mask[None,None,:,:]
        if mask is not None and mask.ndim >= 2 and mask.shape[-2] not in (1,length):
            return original(query,key,value,**kwargs)
        if stats['calls'] == 0:
            print('[H3 query chunks]', 'q='+str(tuple(query.shape)), 'k='+str(tuple(key.shape)),
                  'chunk='+str(chunk), 'score_mib='+str(score_mib), 'full_keys=True',flush=True)
        output = None
        for start in range(0,length,chunk):
            end = min(start+chunk,length)
            options = kwargs.copy()
            options['attn_mask'] = mask
            if mask is not None and mask.ndim >= 2 and mask.shape[-2] == length:
                options['attn_mask'] = mask[...,start:end,:]
            part = original(query[:,start:end,:,:],key,value,**options)
            if output is None:
                output = torch.empty((part.shape[0],length,part.shape[2],part.shape[3]),
                                     device=part.device,dtype=part.dtype)
            output[:,start:end,:,:].copy_(part)
            stats['chunks'] += 1
        stats['calls'] += 1
        return output
    dispatch._h3_query_chunked = True
    dispatch._h3_original = original
    dispatch._h3_stats = stats
    h3.dispatch_attention_fn = dispatch
    return dispatch
