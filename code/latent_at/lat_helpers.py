import torch
import torch.nn.functional as F
from peft import AutoPeftModelForCausalLM, PeftModel

from .utils import *


def compute_toward_away_loss(
    model,
    towards_tokens,
    towards_labels_mask,
    away_tokens,
    away_labels_mask,
    towards_labels,
    away_labels,
    coefs,
    accelerator=None,
):
    # Computes towards_loss + away_loss as defined in HarmBench
    losses = {"total": 0}

    if towards_tokens is not None:
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=towards_tokens).logits
            final_logits = logits[:, :-1][towards_labels_mask[:, 1:]]
            if towards_labels is None:
                towards_labels = towards_tokens[:, 1:][towards_labels_mask[:, 1:]]
            #print("This is final_logits shape", final_logits.shape)
            toward_loss = F.cross_entropy(final_logits, towards_labels)

        if accelerator is not None:
            accelerator.backward(coefs["toward"] * toward_loss)
        else:
            (coefs["toward"] * toward_loss).backward()
        losses["toward"] = toward_loss.item()
        losses["total"] += toward_loss.item()
        
    if away_tokens is not None:
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=away_tokens).logits
            final_logits = logits[:, :-1][away_labels_mask[:, 1:]]
            if away_labels is None:
                away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:]]
            away_loss = log_1_minus_p_loss(final_logits, away_labels)

        if accelerator is not None:
            accelerator.backward(coefs["away"] * away_loss)
        else:
            (coefs["away"] * away_loss).backward()

        losses["away"] = away_loss.item()
        losses["total"] += away_loss.item()

    return losses



def do_adversary_step_padd(
    model,
    batches,
    batch,
    losses_dict,
    coefs,
    log_loss=False,
    wrappers_to_disable_for_reference=[],
    device="cuda",
    accelerator=None,
):
    #breakpoint()
    if "dpo" in coefs: # If running DPO training
        
        toward_tokens = batch["adv_padd_prompt_tokens"].to(device)
        toward_labels_mask = batch["adv_padd_labels_mask"].to(device)
        away_tokens = batch["def_padd_prompt_tokens"].to(device)
        away_labels_mask = batch["def_padd_labels_mask"].to(device)
        
        loss = compute_dpo_loss(
            model=model,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            wrappers_to_disable_for_reference=wrappers_to_disable_for_reference,
            coefs=coefs,
        )
    
    else: # if using another training set up
        
        include_towards_loss = "toward" in coefs and coefs["toward"] > 0
        include_away_loss = "away" in coefs and coefs["away"] > 0
        
        if include_towards_loss:  # a loss for positively supervised behavior
            toward_tokens = batches["adv_padd_prompt_tokens"].to(device)
            toward_labels_mask = batches["adv_padd_labels_mask"].to(device)
            if "adv_labels" in batches:
                if isinstance(batches["adv_labels"], list) and isinstance(batches["adv_labels"][0], list):
                    # flatten the list of lists
                    toward_labels = torch.tensor([item for sublist in batches["adv_labels"] for item in sublist]).to(device)
                else:
                    toward_labels = batches["adv_labels"].to(device)
            else:
                toward_labels = None
        else:
            toward_tokens = None
            toward_labels_mask = None
            toward_labels = None

        if include_away_loss:  # a loss for negatively supervised behavior
            away_tokens = batches["def_padd_prompt_tokens"].to(device)
            away_labels_mask = batches["def_padd_labels_mask"].to(device)
            if "def_labels" in batches:
                # labels is probably a list of lists, check
                if isinstance(batches["def_labels"], list) and isinstance(batches["def_labels"][0], list):
                    away_labels = torch.tensor([item for sublist in batches["def_labels"] for item in sublist]).to(device)
                else:
                    away_labels = batches["def_labels"].to(device)
            else:
                away_labels = None
        else:
            away_tokens = None
            away_labels_mask = None
            away_labels = None

        #breakpoint()

        # compute overall loss
        loss = compute_toward_away_loss(
            model=model,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_labels=toward_labels,
            away_labels=away_labels,
            coefs=coefs,
            accelerator=accelerator,
        )

    # Log loss in dictionary
    if log_loss:
        for key in loss:
            losses_dict["adv_"+key] = loss[key]

def do_adversary_step(
    model,
    batch,
    losses_dict,
    coefs,
    log_loss=False,
    wrappers_to_disable_for_reference=[],
    device="cuda",
    accelerator=None,
):
    #breakpoint()
    if "dpo" in coefs: # If running DPO training
        
        toward_tokens = batch["adv_tokens"].to(device)
        toward_labels_mask = batch["adv_labels_mask"].to(device)
        away_tokens = batch["def_tokens"].to(device)
        away_labels_mask = batch["def_labels_mask"].to(device)
        
        loss = compute_dpo_loss(
            model=model,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            wrappers_to_disable_for_reference=wrappers_to_disable_for_reference,
            coefs=coefs,
        )
    
    else: # if using another training set up
        
        include_towards_loss = "toward" in coefs and coefs["toward"] > 0
        include_away_loss = "away" in coefs and coefs["away"] > 0
        
        if include_towards_loss:  # a loss for positively supervised behavior
            toward_tokens = batch["adv_tokens"].to(device)
            toward_labels_mask = batch["adv_labels_mask"].to(device)
            if "adv_labels" in batch:
                if isinstance(batch["adv_labels"], list) and isinstance(batch["adv_labels"][0], list):
                    # flatten the list of lists
                    toward_labels = torch.tensor([item for sublist in batch["adv_labels"] for item in sublist]).to(device)
                else:
                    toward_labels = batch["adv_labels"].to(device)
            else:
                toward_labels = None
        else:
            toward_tokens = None
            toward_labels_mask = None
            toward_labels = None

        if include_away_loss:  # a loss for negatively supervised behavior
            away_tokens = batch["def_tokens"].to(device)
            away_labels_mask = batch["def_labels_mask"].to(device)
            if "def_labels" in batch:
                # labels is probably a list of lists, check
                if isinstance(batch["def_labels"], list) and isinstance(batch["def_labels"][0], list):
                    away_labels = torch.tensor([item for sublist in batch["def_labels"] for item in sublist]).to(device)
                else:
                    away_labels = batch["def_labels"].to(device)
            else:
                away_labels = None
        else:
            away_tokens = None
            away_labels_mask = None
            away_labels = None

        #breakpoint()

        # compute overall loss
        loss = compute_toward_away_loss(
            model=model,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_labels=toward_labels,
            away_labels=away_labels,
            coefs=coefs,
            accelerator=accelerator,
        )

    # Log loss in dictionary
    if log_loss:
        for key in loss:
            losses_dict["adv_"+key] = loss[key]

