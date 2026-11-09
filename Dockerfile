FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        gnupg \
        wget \
        curl \
        libgomp1 \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends intel-oneapi-mkl-devel=2025.3.0-461 \
    && rm -rf /var/lib/apt/lists/*

# Install tools/depots under /opt, not /root: Apptainer runs as the unprivileged
# host user, who cannot read Docker's /root.
ENV POETRY_HOME=/opt/poetry
ENV PATH="/opt/poetry/bin:${PATH}"
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV JULIA_VERSION=1.12.6
RUN wget -qO /tmp/julia.tar.gz \
      https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf /tmp/julia.tar.gz -C /opt \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia \
    && rm /tmp/julia.tar.gz

WORKDIR /repo
COPY . .

RUN make all

# Pre-installed, read-only depot. Run scripts stack a writable depot in front:
#   JULIA_DEPOT_PATH=/repo/.julia-depot:/opt/julia-depot:
# The trailing ":" keeps Julia's bundled depot (stdlib caches) on the path.
ENV JULIA_DEPOT_PATH=/opt/julia-depot:
RUN bash instantiate_environments.sh && chmod -R a+rX /opt/julia-depot /opt/poetry

CMD ["/bin/bash"]
