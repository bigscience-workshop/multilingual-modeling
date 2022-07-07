for model in tr5b-1B3-multilingual-alpha-checkpoints; do
    for ch in 12000 55500 99000 100500 117000 118500; do
	mname=${model}/ch${ch}
	for dataset in flores ted_multi; do
	    outdir=retrieval_acc_${model}-${dataset}
	    mkdir -p $outdir
	    sbatch compute_retrieval_acc.sh ${mname} ${dataset}
	done
    done
done
