import logging
import os
import re
import traceback

import esm
import runpod
import torch

from interprot.esm_wrapper import ESM2Model
from interprot.sae_model import SparseAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_DIR = "weights"
SAE_NAME_TO_CHECKPOINT = {
    "SAE4096-L24": "esm2_plm1280_l24_sae4096_100Kseqs.pt",
}


def load_models():
    sea_name_to_info = {}
    for sae_name, sae_checkpoint in SAE_NAME_TO_CHECKPOINT.items():
        pattern = r"plm(\d+).*?l(\d+).*?sae(\d+)"
        matches = re.search(pattern, sae_checkpoint)
        if matches:
            plm_dim, plm_layer, sae_dim = map(int, matches.groups())
        else:
            raise ValueError("Checkpoint file must start with plm<n>_l<n>_sae<n>")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ESM2 model
        logger.info(f"Loading ESM2 model with plm_dim={plm_dim}")
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        esm2_model = ESM2Model(
            num_layers=33,
            embed_dim=plm_dim,
            attention_heads=20,
            alphabet=alphabet,
            token_dropout=False,
        )
        esm2_weights = os.path.join(WEIGHTS_DIR, "esm2_t33_650M_UR50D.pt")
        esm2_model.load_esm_ckpt(esm2_weights)
        esm2_model = esm2_model.to(device)

        # Load SAE models (ensure compatibility with Lightning checkpoints)
        logger.info(f"Loading SAE model {sae_name}")
        sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device)
        sae_weights = os.path.join(WEIGHTS_DIR, sae_checkpoint)
        state_dict = torch.load(sae_weights, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("sae_model.", ""): v for k, v in state_dict.items()}
        sae_model.load_state_dict(state_dict)

        sea_name_to_info[sae_name] = {
            "model": sae_model,
            "plm_layer": plm_layer,
        }

    logger.info("Models loaded successfully")
    return esm2_model, sea_name_to_info


def handler(event):
    try:
        input_data = event["input"]
        seq = input_data["sequence"]
        sae_name = input_data["sae_name"]
        dim = input_data["dim"]
        multiplier = input_data["multiplier"]
        logger.info(f"sae_name: {sae_name}, dim: {dim}, multiplier: {multiplier}")

        sae_info = sea_name_to_info[sae_name]
        sae_model = sae_info["model"]
        plm_layer = sae_info["plm_layer"]

        # First, get ESM layer 24 activations, encode it with SAE to get a (L, 4096) tensor
        _, esm_layer_acts = esm2_model.get_layer_activations(seq, plm_layer)
        sae_latents, mu, std = sae_model.encode(esm_layer_acts[0])

        # Decode the SAE latents yields a (L, 1280) tensor `decoded_esm_layer_acts`,
        # i.e. the SAE's prediction of ESM layer 24 acts. Compute the error as `recons_error`.
        esm_layer_acts_dec = sae_model.decode(sae_latents, mu, std)
        recons_error = esm_layer_acts - esm_layer_acts_dec

        # Steer by setting the latent dim activation of it's max activation * multiplier
        base_act = sae_latents.max() if multiplier > 0 else sae_latents.min()
        sae_latents[:, dim] = base_act * multiplier

        # Decode with modified SAE latents and add back the reconstruction error
        steered_esm_layer_acts_dec = sae_model.decode(sae_latents, mu, std)
        logits = esm2_model.get_sequence((steered_esm_layer_acts_dec + recons_error), 24)

        # Take argmax over the logits to get the steered sequence
        steered_tokens = torch.argmax(logits[0, 1:-1, 4:24], dim=-1)
        steered_sequence = "".join([esm2_model.alphabet.all_toks[i + 4] for i in steered_tokens])

        return {
            "status": "success",
            "data": {
                "steered_sequence": steered_sequence,
            },
        }
    except Exception as e:
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


esm2_model, sea_name_to_info = load_models()
runpod.serverless.start({"handler": handler})