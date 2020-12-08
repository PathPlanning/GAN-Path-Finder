#!/bin/bash

# obligatory docker/singularity container namings
export PROJECT="gan_finder"
export SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
export IMAGE_NAME="nsob/${PROJECT}"
export IMAGE_VERSION="base"
#export IMAGE_VERSION="meshcnn"
#export IMAGE_VERSION="graphstar"
export IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
export SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

