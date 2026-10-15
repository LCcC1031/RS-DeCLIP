# Algorithm pseudocode

## Stage 1: dual-source adaptation

```text
Inputs: RemoteCLIP-compatible dual encoder, raw/polished training captions
Build clipped BM25-IDF from the training captions
Initialize predictor, text-neighbor memory, cross-attention head, and ADCA head

For each training batch:
    Sample two image views for each image
    Sample one polished caption and one raw caption of that image
    Construct a counterfactual caption by replacing one supported detail phrase
    Encode global image/text features and the first view's patch/token features
    Compute image-text contrastive and three cross-view contrastive objectives
    Downweight highly similar semantic negatives in the configured objectives
    Compute image self-supervision through the predictor
    Compute text-view consistency weight = stop_gradient(((cos(text1,text2)+1)/2)^2)
    Apply this weight to text self-supervision
    Read nearest text features from the memory and compute neighbor supervision
    Compute auxiliary cross-attention alignment
    ADCA: score each token by its top-3 image-patch similarities
          weight tokens using clipped IDF and a learned gate
          compare matched and permuted image detail scores
    Compute the valid counterfactual caption ranking objective
    Combine the original weighted objectives and update trainable parameters
    Update the text-neighbor memory after the optimizer step

Inference: normalized image embedding dot normalized text embedding
```

## Stage 2: five-query P2-CGVF

```text
Inputs: frozen Stage-1 encoders; pretrained R50-FPN; raw training captions
Freeze R50; train the FPN and query/projection/fusion modules
Initialize five trainable queries and anchor offsets
Initial anchors in (vertical,horizontal) coordinates:
    (0.5,0.5), (0.2,0.2), (0.2,0.8), (0.8,0.2), (0.8,0.8)

For each image:
    g = normalized frozen Stage-1 image embedding
    P2 = first R50-FPN feature map
    q = LayerNorm(query) + 0.15 * LayerNorm(project(g))
    anchors = initial_anchors + 0.10 * tanh(offsets)
    attention = softmax(Q*K^T/sqrt(head_dim) - squared_distance/(2*0.35^2))
    tokens = LayerNorm(output_projection(attention*V))
    local_tokens = normalize(shared_MLP(tokens))
    weights = softmax(cos(local_tokens,g)/0.07)
    r = weighted_sum(local_tokens)
    r = normalize(r + 0.10*refinement_MLP(r))
    consistency = cos(r,g)
    orthogonal_residual = r - consistency*g
    gate = learned_bounded_amplitude * sigmoid((consistency-learned_center)/learned_temperature)
    final = normalize(g + gate*orthogonal_residual)

For each training batch:
    Encode all selected images' raw captions with the frozen text encoder
    Build the text-to-image positive mask from image identities
    During epochs 1 and 2, blend global and final scores with min(1,epoch/2)
    Compute the original multi-positive image-text retrieval objective
    Add global alignment, local-token alignment, token diversity, and attention overlap terms
    Update the trainable FPN and local modules

Inference: normalized text embedding dot final image descriptor
```

