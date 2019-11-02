#!/usr/bin/env bash

# If project root not defined.
if [[ -z "$DIR_TMP" ]]; then    
    # get the directory of this file.
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/.."; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"


# check if data exist.
if [[ ! -d ${DIR_TMP}/train ]]; then
    echo "Downloading data......"
    # download trian and test data.
    wget https://doc-10-ao-docs.googleusercontent.com/docs/securesc/fkeaflsv8c4v1eieuq850ii03dmi0bdc/guq0eu8pl6kugons6hrs0oflha3l47d6/1572660000000/01812282273416792731/01812282273416792731/1oJ1QRwEF_FywXEAfxYlzHaPSSV_qbSKq?e=download
    wget https://doc-0c-ao-docs.googleusercontent.com/docs/securesc/fkeaflsv8c4v1eieuq850ii03dmi0bdc/4auais6stn2m67pjdupkojch75s7daf3/1572660000000/01812282273416792731/01812282273416792731/1mPtpqD51pW3OsmBL_WXYrXRqQI5FQtv8?e=download&nonce=47lafemg54n0k&user=01812282273416792731&hash=p76b7qfb7hvn355e19n3rgv982pm2im0
    unzip "train.zip" && rm "train.zip" && rm "__MACOSX"
    unzip "test.zip" && rm "test.zip" && rm "__MACOSX"
fi

echo "Data Downloading Completed"
