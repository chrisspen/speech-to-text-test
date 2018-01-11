#!/bin/bash
pylint --version
echo "Running pylint..."
pylint --rcfile=pylint.rc src
