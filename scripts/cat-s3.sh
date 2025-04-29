#!/bin/bash

set -Eeu -o pipefail -x

bucket="$1"
key="$2"
local_file_to_concatenate="$3"
local_copy=$(mktemp)
local_concatenated=$(mktemp)
n_failures=0

while (( n_failures < 2 )); do
    current_etag=$(aws s3api head-object --bucket "$bucket" --key "$key" --query ETag --output text)
    if [[ "$current_etag" == "null" ]]; then
        echo "Failed to retrieve ETag. Exiting."
        exit 1
    fi

    aws s3api get-object --bucket "$bucket" --key "$key" --if-match "$current_etag" "$local_copy" || {
        echo "ETag does not match. Trying again."
        n_failures=$(( n_failures + 1 ))
        continue
    }

    if [[ "$key" =~ \.gz$ ]]; then
        local_decompressed=$(mktemp)
        local_decompressed_concat=$(mktemp)
        gzip -d -c "$local_copy" > "$local_decompressed"
        cat "$local_decompressed" "$local_file_to_concatenate" > "$local_decompressed_concat"
        gzip -c "$local_decompressed_concat" > "$local_concatenated"
        rm "$local_decompressed" "$local_decompressed_concat"
    else
        cat $local_copy $local_file_to_concatenate > $local_concatenated
    fi


    aws s3api put-object --bucket "$bucket" --key "$key" --body "$local_concatenated" --if-match "$current_etag" || {
        echo "ETag does not match during upload. Trying again."
        n_failures=$(( n_failures + 1 ))
        continue
    }

    echo "File updated and uploaded successfully."
    exit 0
done

echo "Too many failures: $n_failures."
exit 1
