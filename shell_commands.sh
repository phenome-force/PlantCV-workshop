#!/bin/bash

plantcv-workflow.py --config sorghum_morphology_workflow_config.json

plantcv-utils.py json2csv --json sorghum_results.json --csv sorghum_subset_data

