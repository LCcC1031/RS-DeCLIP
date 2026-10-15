"""Run with: python -m examples.core_usage"""

import torch
from torch.nn import functional as F

from rs_declip import P2QueryCGVF, stage2_objective


def main():
    image_count = 2
    captions_per_image = 5
    p2 = torch.randn(image_count, 256, 8, 8, requires_grad=True)
    global_features = F.normalize(torch.randn(image_count, 512), dim=-1)
    text_features = F.normalize(torch.randn(image_count * captions_per_image, 512), dim=-1)
    caption_to_image = torch.arange(image_count).repeat_interleave(captions_per_image)
    positive_mask = caption_to_image[:, None] == torch.arange(image_count)[None, :]

    module = P2QueryCGVF()
    local_tokens, descriptor, info = module(p2, global_features)
    loss, _ = stage2_objective(
        local_tokens, descriptor, global_features, text_features,
        positive_mask, torch.tensor(1.0 / 0.07), info["query_overlap_loss"], epoch=1,
    )
    loss.backward()
    print("Local tokens:", tuple(local_tokens.shape))
    print("Retrieval descriptor:", tuple(descriptor.shape))
    print("Text-to-image scores:", tuple((text_features @ descriptor.t()).shape))
    print("Loss is finite:", bool(torch.isfinite(loss)))


if __name__ == "__main__":
    main()
