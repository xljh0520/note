padding sequence
```
torch.nn.utils.rnn.pad_sequence(
    [d["caption_tokens"] for d in data], # list of tensor with shape (len, ) len is not the same
    batch_first=True,
    padding_value=padding_idx,
)
###
x = torch.tensor([2,3,4,5,5]).long()
y = torch.tensor([3,4,5]).long()
z = torch.tensor([1,2]).long()
padding_idx = -1
torch.nn.utils.rnn.pad_sequence(
    [x, y, z], # list of tensor with shape (len, ) len is not the same
    batch_first=True,
    padding_value=padding_idx,
)
##
tensor([[ 2,  3,  4,  5,  5],
        [ 3,  4,  5, -1, -1],
        [ 1,  2, -1, -1, -1]])
```

length tensor -> mask tensor
```
# caption_tokens: (B, L)
# caption_lengths: (B,)
ones = torch.ones_like(caption_tokens)
caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)
# caption_mask: (B, L) [[False, False, False,  True,  True,  True,  True,  True,  True,  True]]
```

caption unidirectional mask 
```
def _generate_future_mask(
         size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
    r"""
    Generate a mask for "future" positions, useful when using this module
    for language modeling.
    """
    # Default mask is for forward direction. Flip for backward direction.
    mask = torch.triu(
        torch.ones(size, size, device=device, dtype=dtype), diagonal=1
    )
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask
###
_generate_future_mask(size=10, dtype=torch.float32, device='cpu')

tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```
