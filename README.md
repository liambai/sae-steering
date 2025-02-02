# SAE Steering Endpoint on RunPod

## Setup and testing

Build and run a Docker container

```bash
docker compose build
docker compose run --rm sae-steering bash
```

Test the endpoint (make sure Docker is allocated ~6G of memory)

```bash
python3 handler.py --test_input '{"input": {"sae_name": "SAE4096-L24", "sequence": "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN", "dim": 220, "multiplier": 1}}'
```
