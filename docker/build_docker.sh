#!/bin/bash

set -e

# example launch string:
# ./build_docker.sh [-p] [-v]
#     -p:       push the build image to the dockerhub under 'artonson' username
#     -v:       be verbose

usage() { echo "Usage: $0 [-f dockerfile] [-p] [-v]" >&2; }

DOCKERFILE=Dockerfile.base
VERBOSE=false
PUSH=false
while getopts "f:pv" opt
do
    case ${opt} in
        f) DOCKERFILE=${OPTARG};;
        p) PUSH=true;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
fi

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"
source "${PROJECT_ROOT}"/env.sh

# overwrite IMAGE_VERSION
IMAGE_VERSION="${DOCKERFILE##*.}"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"

DOCKERFILE="${PROJECT_ROOT}/docker/${DOCKERFILE}"     # full pathname of Dockerfile.base

echo "******* BUILDING IMAGE ${IMAGE_NAME}, version ${IMAGE_VERSION} *******"

docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME_TAG}" \
    "${PROJECT_ROOT}"


if [[ "${PUSH}" = true ]]; then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push "${IMAGE_NAME_TAG}"
fi
