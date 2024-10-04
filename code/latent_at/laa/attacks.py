import torch
import einops
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import random

def find_nearest_indices_and_similarities(tensor1, tensor2):
    """
    Calculate cosine similarity between two tensors and find nearest indices.
    
    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
    
    Returns:
        tuple: Indices and similarities.
    """
    tensor1_norm = F.normalize(tensor1, dim=1)
    tensor2_norm = F.normalize(tensor2, dim=1)
    
    similarity_matrix = torch.mm(tensor1_norm, tensor2_norm.t())
    similarities, indices = torch.max(similarity_matrix, dim=1)
    
    return indices.float().cpu().tolist(), similarities.float().cpu().tolist()

class LowRankAdversary(nn.Module):
    """Low-rank adversary model."""
    
    def __init__(self, dim, rank, device, bias=False, zero_init=True):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False).to(device)
        self.lora_B = nn.Linear(rank, dim, bias=bias).to(device)
        if zero_init:
            self.lora_B.weight.data.zero_()
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) + x

class FullRankAdversary(nn.Module):
    """Full-rank adversary model."""
    
    def __init__(self, dim, device, bias=False):
        super().__init__()
        self.m = nn.Linear(dim, dim, bias=bias).to(device)
        self.m.weight.data.zero_()
    
    def forward(self, x):
        return self.m(x) + x

def get_next_available_path(base_path, base_name, extension=""):
    """Get the next available file path."""
    index = 1
    while True:
        path = os.path.join(base_path, f"{base_name}_{index}{extension}.npy")
        if not os.path.exists(path):
            return path
        index += 1

def get_new_path():
    """Get a new path for temporary attack data."""
    return get_next_available_path("/root/autodl-tmp/at_final/uap", "temp_attack")

def get_new_x_path():
    """Get a new path for temporary x data."""
    return get_next_available_path("/root/autodl-tmp/at_final/uap", "temp_x")

def compute_toward_away_loss_adversary(model, towards_tokens, towards_labels_mask, away_tokens, away_labels_mask, towards_labels=None, away_labels=None, coefs={"toward": 1.0, "away": 1.0}, accelerator=None):
    """Compute the toward and away losses for the adversary."""
    losses = {"total": 0}

    if towards_tokens is not None:
        with torch.no_grad():
            logits = model(input_ids=towards_tokens.to("cuda:0")).logits
            final_logits = logits[:, :-1][towards_labels_mask[:, 1:]]
            if towards_labels is None:
                towards_labels = towards_tokens[:, 1:][towards_labels_mask[:, 1:]].to("cuda:0")
            toward_loss = F.cross_entropy(final_logits, towards_labels)
            losses["toward"] = toward_loss.item()
            losses["total"] += toward_loss.item()

    if away_tokens is not None:
        with torch.no_grad():
            logits = model(input_ids=away_tokens.to("cuda:0")).logits
            final_logits = logits[:, :-1][away_labels_mask[:, 1:]]
            if away_labels is None:
                away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:]].to("cuda:0")
            away_loss = F.cross_entropy(F.log_softmax(final_logits, dim=-1), away_labels)
            losses["away"] = away_loss.item()
            losses["total"] += away_loss.item()

    return losses

class GDAdversary(nn.Module):
    """Gradient Descent Adversary model."""
    
    def __init__(self, dim, epsilon, attack_mask, batch, batches, embedding_space=None, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.batch = batch
        self.batches = batches
        self.attack_mask = attack_mask
        self.save_flag = False
        
        if dtype:
            self.attack = nn.Parameter(torch.zeros(attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device, dtype=dtype))
        else:
            self.attack = nn.Parameter(torch.zeros(attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        
        self.embedding_space = torch.tensor(embedding_space, device=self.device, dtype=self.attack.dtype)
        self.embedding_space.requires_grad_(False)
        
        random_indices = torch.randint(0, self.embedding_space.shape[0], (1, attack_mask.shape[1]))
        initial_attack = self.embedding_space[random_indices].to(device)
        self.attack = nn.Parameter(initial_attack) 
        
        uap_path = "/root/autodl-tmp/at_final/success_multi/success_uap.npy"
        if os.path.exists(uap_path):
            uap_tensor = torch.tensor(np.load(uap_path), device=self.device, dtype=self.attack.dtype)
            new_attack = self.attack.clone()
            new_attack[self.attack_mask] = uap_tensor
            self.attack = nn.Parameter(new_attack)    

        self.clip_attack()
        
    def forward(self, x):
        if x.shape[0] == 1:
            if x.shape[1] == 1 and self.attack.shape[1] != 1:
                return x
            else:
                if self.device is None or self.device != x.device:
                    with torch.no_grad():
                        self.device = x.device
                        self.attack.data = self.attack.data.to(self.device)
                        self.attack_mask = self.attack_mask.to(self.device)
                
                if x.shape[1] == self.attack_mask.shape[1]:
                    perturbed_acts = self.attack[self.attack_mask[:, :x.shape[1]]].to(x.dtype)
                    x[self.attack_mask[:, :x.shape[1]]] = perturbed_acts
                elif x.shape[1] != self.attack_mask.shape[1] and x.shape[1] != 1:
                    temp_mask = self.attack_mask[:, :x.shape[1]]
                    temp_attack = self.attack[:, :x.shape[1], :]
                    perturbed_acts = temp_attack[temp_mask[:, :x.shape[1]]].to(x.dtype)
                    x[temp_mask[:, :x.shape[1]]] = perturbed_acts
                    if self.save_flag:
                        indices, proba = find_nearest_indices_and_similarities(x[0], self.embedding_space)
                        print(proba)
                        print(np.array(indices).astype(int) + 1)
                        self.save_flag = False
                        temp_x_path = get_new_x_path()
                        np.save(temp_x_path, x.to(torch.float32).detach().cpu().numpy())
                        print("save to", temp_x_path)
                return x
        else:
            if x.shape[1] == 1 and self.attack.shape[1] != 1:
                return x
            else:
                if self.device is None or self.device != x.device:
                    with torch.no_grad():
                        self.device = x.device
                        self.attack.data = self.attack.data.to(self.device)
                        self.attack_mask = self.attack_mask.to(self.device)
                
                perturbed_acts = self.attack[self.attack_mask[:, :x.shape[1]]].to(x.dtype)
                attack_masks = self.batches["padd_prompt_mask"]
                for i in range(x.shape[0]):
                    true_positions = torch.nonzero(attack_masks[i], as_tuple=True)[0]
                    x[i, true_positions] = perturbed_acts

                return x

    def save_uap(self, path, logger, shape_1=0):
        temp_mask = self.attack_mask[:, :shape_1]
        temp_attack = self.attack[:, :shape_1, :]
        token_embedding = temp_attack[temp_mask[:, :shape_1]]
        np.save(path, temp_attack[temp_mask[:, :shape_1]].to(torch.float32).detach().cpu().numpy())

        indices, proba = find_nearest_indices_and_similarities(token_embedding, self.embedding_space)
        print(proba)
        logger.info("save current uap")
        logger.info(np.array(indices)+1)
        logger.info(proba)
        
    def calculate_top_k_embedding_loss(self, k=10):
        loss = 0.0
        for i in range(self.attack.shape[1]):
            distances = torch.norm(self.embedding_space - self.attack[0, i], dim=-1)
            top_k_indices = torch.topk(distances, k, largest=False).indices
            top_k_embeddings = self.embedding_space[top_k_indices]
            
            cosine_similarity = nn.CosineSimilarity(dim=-1)
            similarity_scores = cosine_similarity(self.attack[0, i].unsqueeze(0), top_k_embeddings)
            
            top_k_loss = -torch.mean(similarity_scores)
            loss += top_k_loss    
        return loss

    def calculate_embedding_loss(self):
        loss = 0.0
        for i in range(self.attack.shape[1]):
            cosine_similarity = nn.CosineSimilarity(dim=-1)
            similarity_scores = cosine_similarity(self.attack[0, i].unsqueeze(0), self.embedding_space)
            
            total_loss = -torch.mean(similarity_scores)
            loss += total_loss    
        return loss
    
    def _compute_losses(self, candidate_embeddings, model, index):
        original_attack = self.attack.detach().clone()
        self.attack[0, index] = candidate_embeddings.to(self.device)
        with torch.no_grad():
            losses = compute_toward_away_loss_adversary(
                model,
                self.batch["adv_padd_prompt_tokens"],
                self.batch["adv_padd_labels_mask"],
                self.batch["def_padd_prompt_tokens"],
                self.batch["def_padd_labels_mask"],
                self.batch.get("adv_labels"),
                self.batch.get("def_labels"),
                {"toward": 1.0, "away": 1.0},
            )
            total_loss = losses["total"]

        self.attack[0] = original_attack
        del original_attack

        return total_loss

    def select_best_embedding(self, k=10, model=None):
        with torch.no_grad():
            best_embeddings = self.attack.clone()
            attack_indices = self.attack_mask[0].nonzero(as_tuple=True)[0]
            for i in attack_indices:
                print(i)
                distances = torch.norm(self.embedding_space - self.attack[0, i], dim=-1)
                top_k_indices = torch.topk(distances, k, largest=False).indices
                top_k_embeddings = self.embedding_space[top_k_indices]
                total_losses = torch.tensor([
                    self._compute_losses(candidate.unsqueeze(0), model=model, index=i) for candidate in top_k_embeddings
                ], device=self.device)
                
                best_idx = torch.argmin(total_losses)
                best_embeddings[0, i] = top_k_embeddings[best_idx]
                del distances, top_k_indices, top_k_embeddings, total_losses
            
            self.attack.copy_(best_embeddings)
            del best_embeddings

    def compute_away_loss(self, model):
        away_tokens = self.batch["def_tokens"].to(self.device)
        away_labels_mask = self.batch["def_labels_mask"].to(self.device)
        logits = model(input_ids=away_tokens).logits
        final_logits = logits[:, :-1][away_labels_mask[:, 1:].to(self.device)]
        away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:].to(self.device)]
        loss = nn.CrossEntropyLoss()(final_logits, away_labels)
        del final_logits, away_labels, logits
        return loss.item()

    def compute_toward_loss(self, model):
        toward_tokens = self.batch["adv_tokens"].to(self.device)
        toward_labels_mask = self.batch["adv_labels_mask"].to(self.device)
        logits = model(input_ids=toward_tokens).logits
        final_logits = logits[:, :-1][toward_labels_mask[:, 1:].to(self.device)]
        toward_labels = toward_tokens[:, 1:][toward_labels_mask[:, 1:].to(self.device)]
        loss = nn.CrossEntropyLoss()(final_logits, toward_labels)
        del final_logits, toward_labels, logits
        return loss.item()

    def clip_attack(self, select_best=False, model=None):
        if select_best and model is not None:
            print("new clip attack")
            self.select_best_embedding(k=5, model=model)
        else:
            if self.embedding_space is None:
                with torch.no_grad():
                    norms = torch.norm(self.attack, dim=-1, keepdim=True)
                    scale = torch.clamp(norms / self.epsilon, min=1)
                    self.attack.div_(scale)
                    norms = torch.norm(self.attack, dim=-1)
            else:
                print("start clipping")
                with torch.no_grad():
                    closest_embeddings = torch.zeros_like(self.attack)
                    for i in range(self.attack.shape[1]):
                        distances = torch.norm(self.embedding_space - self.attack[0, i], dim=-1)
                        top3_indices = torch.topk(distances, k=3, largest=False).indices
                        chosen_idx = random.choice(top3_indices.tolist())
                        closest_embeddings[0, i] = self.embedding_space[chosen_idx]
                    self.attack.copy_(closest_embeddings)

class WhitenedGDAdversary(nn.Module):
    """Whitened Gradient Descent Adversary model."""
    
    def __init__(self, dim, device, epsilon, attack_mask, proj=None, inv_proj=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.proj = proj
        self.inv_proj = torch.inverse(proj) if inv_proj is None else inv_proj
        self.attack = nn.Parameter(torch.randn(attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        self.clip_attack()
        self.attack_mask = attack_mask
    
    def forward(self, x):
        unprojected_attack = torch.einsum("n d, batch seq n-> batch seq d", self.inv_proj, self.attack)
        x[self.attack_mask[:, :x.shape[1]]] = (x + unprojected_attack.to(x.dtype))[self.attack_mask[:, :x.shape[1]]]
        return x
    
    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)
            norms = torch.norm(self.attack, dim=-1)