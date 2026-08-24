import logging

from huggingface_hub import create_branch, create_repo, get_token, upload_file, whoami

logger = logging.getLogger(__name__)


def resolve_token(explicit_token: str = "") -> str:
    token = explicit_token or get_token()
    if not token:
        raise ValueError("Hugging Face token is required. Pass --hf_token or run huggingface-cli login.")
    return token


def ensure_dataset_branch(repo_id: str, branch: str, token: str) -> None:
    logger.info("Using Hugging Face user %s.", whoami(token=token).get("name", "unknown"))
    create_repo(repo_id, token=token, private=True, repo_type="dataset", exist_ok=True)
    create_branch(repo_id, branch=branch, token=token, repo_type="dataset", exist_ok=True)


def upload_parquet(
    output_parquet_file_path: str,
    path_in_repo: str,
    repo_id: str,
    revision: str,
    token: str,
    parquet_idx: int,
) -> None:
    upload_file(
        path_or_fileobj=output_parquet_file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload filtered dataset {parquet_idx}",
        revision=revision,
    )

