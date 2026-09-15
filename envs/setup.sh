#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

git -C Finch.jl fetch origin --tags desc

[ -d Finch-wingspan.jl ] || git -C Finch.jl worktree add ../Finch-wingspan.jl wingspan-arxiv-v0
[ -d Finch-desc.jl ] || git -C Finch.jl worktree add ../Finch-desc.jl origin/desc

julia --project=envs/wingspan -e 'using Pkg; Pkg.instantiate()'
julia --project=envs/desc -e 'using Pkg; Pkg.instantiate()'
