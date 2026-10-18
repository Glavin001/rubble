# The Modal CLI, pinned, for engine/tools/modal/modal.sh: keeps Python and
# Modal off the host. Only drives Modal; builds and GPU runs happen remotely.
FROM python:3.12-slim
RUN pip install --no-cache-dir modal==1.6.2
WORKDIR /repo
ENTRYPOINT ["modal"]
