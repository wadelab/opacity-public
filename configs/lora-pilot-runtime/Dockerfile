# Honest-LoRA pilot runtime image (docs/viking-lora-runner.md). Built with
# Docker on d2-1 and converted to a SIF on Viking by
# scripts/lora_pilot_image.sh. The build context holds this file, the hashed
# lock and Viking's uv binary (sha256 recorded in the runtime lock).
FROM debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8
COPY uv /usr/local/bin/uv
COPY requirements.lock /opt/requirements.lock
RUN export UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 \
 && uv python install 3.12.13 \
 && uv venv --python 3.12.13 /opt/venv \
 && uv pip sync --python /opt/venv/bin/python --require-hashes /opt/requirements.lock \
 && mkdir -p /bundle /model /previous /home/pilot
