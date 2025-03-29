#! /bin/bash
help_str="
  -h, --help:           print helo info 
  -g, --genome string:  reference chromsizes
  -o, --outdir string:         output directory
"
getopt_cmd=$(getopt -o hg:b:o: --long help,genome:,bedfiles:,outdir: -n $(basename $0) -- "$@")
[ $? -ne 0 ] && exit 1
eval set -- "$getopt_cmd"

while [ -n "$1" ]
do
    case "$1" in
        -h|--help)
            echo -e "$help_str"
            exit ;;
        -g|--genome)
            genome="$2"
            shift ;;
        -o|--outdir)
            outdir="$2"
            shift ;;
        --) shift
            break ;;
         *) echo "$1 is not an invalid option"
            exit 1 ;;
    esac
    shift
done
dir=`pwd`
cd ${outdir}
arr=($*)
bedtools makewindows -g $genome -w 200 -s 200 > hg19.txt
mkdir -p tmp/
for i in ${arr[@]}
do
	cp ${dir}/${i} tmp/
done	

bedtools multiinter -i tmp/* | bedtools window -w 900 -u -a hg19.txt -b - > overlap
cp overlap coverage
for((i=0;i<${#arr[*]};i++))
do
echo 'Processing' ${arr[i]}
bedtools coverage -a overlap -b ${dir}/${arr[i]} | cut -f 5 -| paste coverage - > tmp_cov;
mv tmp_cov coverage;
done
rm -r tmp/
rm overlap 
