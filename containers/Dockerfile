# Digest-pinned CPU Python base and exact Python package versions.
ARG PYTHON_BASE=python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
FROM ${PYTHON_BASE}
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COPY locks/python-cpu.txt /opt/drugperturb/python-cpu.txt
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /opt/drugperturb/python-cpu.txt && python -m pip check
WORKDIR /work
# Source/data are staged by Nextflow. This CPU core image intentionally has no R.
CMD ["python", "--version"]
