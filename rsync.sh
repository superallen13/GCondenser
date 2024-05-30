#!/bin/zsh

usrid='uqyliu71'
ignore_file='.rsyncignore'
local_path='./'

remote_path_wiener="uqyliu71@wiener.hpc.dc.uq.edu.au:/scratch/itee/uqyliu71/GCondenser-syn"
remote_path_bunya="uqyliu71@bunya.rcc.uq.edu.au:/scratch/user/uqyliu71/GCondenser-syn"
remote_path=$remote_path_wiener

while getopts "t:" opt; do
    case ${opt} in
    t)
        target=$OPTARG
        ;;
    \?)
        echo "Usage: cmd [-t target]"
        exit 1
        ;;
    esac
done

if [ "$target" == "wiener" ]; then
    remote_path=$remote_path_wiener
elif [ "$target" == "bunya" ]; then
    remote_path=$remote_path_bunya
else
    echo "Invalid target specified. Use 'wiener' or 'bunya'."
    exit 1
fi

rsync -a -v --delete --exclude-from $ignore_file $local_path $remote_path

echo "Transfer to $target complete."
