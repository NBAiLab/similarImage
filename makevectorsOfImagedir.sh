#!/bin/bash
if test $# -ne  2
then
echo "Usage: $0 <Imagedir> <vector dir>"
exit
fi

for i in `find $1 -name "*.jpg"`
do
  echo $i
  bname=`basename $i | cut -d . -f 1`
  python3 makeImageVector.py $i $2/${bname}.npz


done
