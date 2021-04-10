#!/bin/bash
# build package and upload to pypi

if [ $# -ne 1 ]
then
    echo "only testing package build"
    PUBLISH_TYPE=testing
else
    PUBLISH_TYPE=$1
fi

rm dist/*

python setup.py sdist bdist_wheel

twine check dist/*

if [ $? -ne 0 ]
then
    fail "Package did not build successfully"
    exit 1
fi

if [ "$PUBLISH_TYPE" == publish ]
then
    twine upload dist/*
fi
