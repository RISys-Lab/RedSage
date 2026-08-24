# RedSage Data

This folder contains data-processing modules for RedSage. The root
[`README.md`](../README.md) summarizes the released datasets and links to the public dataset collection; module-specific READMEs provide setup and usage instructions.

## Modules

| Module | Status | Purpose |
| --- | --- | --- |
| [`FineWebSecurity`](FineWebSecurity/README.md) | Available | Filters FineWeb with the cybersecurity BERT/ModernBERT classifier to build CyberFineWeb-style cybersecurity web data. The released filtered corpus is available as [`RISys-Lab/RedSage-CFW`](https://huggingface.co/datasets/RISys-Lab/RedSage-CFW). |
| `AgenticDataAugmentation` | Coming soon | Generates and validates multi-turn, role-grounded cybersecurity dialogues from seed data. |

## Notes

- `FineWebSecurity` is self-contained and optional; its dependencies are not required for RedSage inference, training, or evaluation.
- Legacy API, vLLM, prompt-based, and embedding FineWeb filters were removed. The available FineWeb workflow is the BERT/ModernBERT filtering path.
- AgenticDataAugmentation will contain the RedSage-Conv generation pipeline when it is released.

For FineWebSecurity installation, filtering commands, output layout, resume behavior, Hugging Face upload, and Docker usage, see [`FineWebSecurity/README.md`](FineWebSecurity/README.md).
