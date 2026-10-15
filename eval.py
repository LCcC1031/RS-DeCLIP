"""Single-checkpoint RemoteCLIP / Stage-1 retrieval evaluation.

The original test split, capitalization, prompt, grouping, and recall calculation
are retained. This entry point evaluates the dual encoder without a Stage-2 branch.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

RS_PROMPT = "An aerial satellite image of "


def get_model(args):
    import clip

    model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    checkpoint = torch.load(args.remoteclip_path, map_location=args.device, weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, preprocess, preprocess, preprocess, clip.tokenize


class JsonDataset(Dataset):

    def __init__(self, json_dir, img_dir, transforms):
        self.json_dir = json_dir
        self.transforms = transforms
        self.img_dir = img_dir
        self.images = []
        self.captions = []
        self.read_json()
        self.duplicate()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = Image.open(os.path.join(self.img_dir, self.images[idx]))
        images = self.transforms(images)
        texts = self.captions[idx]
        return (images, texts)

    def read_json(self):
        with open(self.json_dir, 'r', encoding='utf-8-sig') as stream:
            datasets = json.load(stream)
        for image in datasets['images']:
            if image['split'] == 'test':
                for text in image['sentences']:
                    self.images.append(image['filename'])
                    self.captions.append(text['raw'].capitalize())

    def duplicate(self):
        unique_images, indexs = np.unique(self.images, return_index=True)
        if len(unique_images) != len(self.images):
            self.duplicated_images = []
            self.duplicated_captions = []
            for index in indexs:
                self.duplicated_images.append(self.images[index])
                same_indexs = [i for i, x in enumerate(self.images) if x == self.images[index]]
                captions = []
                for same_index in same_indexs:
                    captions.append(self.captions[same_index])
                self.duplicated_captions.append(captions)
            self.images = self.duplicated_images
            self.captions = self.duplicated_captions

def retrieval_evaluation(args, model, preprocess, tokenize, recall_k_list=(1, 5, 10)):
    from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
    from clip_benchmark.datasets.builder import get_dataset_collate_fn
    dataset = JsonDataset(args.retrieval_json_dir, args.retrieval_images_dir, preprocess)
    if len(dataset) == 0:
        raise ValueError('No test images were found in the original annotation JSON.')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=get_dataset_collate_fn('mscoco_captions'))
    n_batches = len(dataloader)
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    for batch_images, batch_texts, inds in tqdm(dataloader, total=n_batches, desc='Encoding features'):
        batch_images = batch_images.to(args.device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        batch_texts = tokenize([RS_PROMPT + text for i, texts in enumerate(batch_texts) for text in texts]).to(args.device)
        with torch.no_grad():
            batch_image_features = model.encode_image(batch_images)
            batch_text_features = model.encode_text(batch_texts)
            batch_images_emb = F.normalize(batch_image_features, dim=-1)
            batch_texts_emb = F.normalize(batch_text_features, dim=-1)
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
    batch_size = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        metrics[f'retrieval-image2text-R@{recall_k}'] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, args.device, k=recall_k) > 0).float().mean().item() * 100
    for recall_k in recall_k_list:
        metrics[f'retrieval-text2image-R@{recall_k}'] = (batchify(recall_at_k, scores, positive_pairs, batch_size, args.device, k=recall_k) > 0).float().mean().item() * 100
    metrics[f'retrieval-mean-recall'] = np.mean(list(metrics.values()))
    for key, item in metrics.items():
        metrics[key] = round(float(item), 2)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", dest="remoteclip_path", required=True)
    parser.add_argument("--json-path", dest="retrieval_json_dir", required=True,
                        help="Original raw-caption JSON containing the test split.")
    parser.add_argument("--image-root", dest="retrieval_images_dir", required=True)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be nonnegative.")
    model, _, _, preprocess, tokenize = get_model(args)
    metrics = retrieval_evaluation(args, model, preprocess, tokenize)
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()

