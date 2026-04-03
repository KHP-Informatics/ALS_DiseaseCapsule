#!/bin/bash -l
#SBATCH --mem=983G
i=$1
file=$2
#python chi2_test.allele.py sNL.vcf sNL.txt
#python all_pca.py --vcf sNL.vcf --labels_file labels.csv
perl  /scratch/prj/bcn_ml_als/annovar/convert2annovar.pl -format vcf4old  /scratch/prj/bcn_ml_als/vcf_imputed/$file.$i.vcf > chr$i.avinput 2>/dev/null 
perl /scratch/prj/bcn_ml_als/annovar/annotate_variation.pl -out chr$i -build hg19 chr$i.avinput  /scratch/prj/bcn_ml_als/annovar/humandb
cat chr$i.variant_function|perl -ne 'my@a=split;my$gene; if ($a[0] eq "intergenic"){my@aaa=$a[1]=~/(\S+)\(dist=(\S+)\),(\S+)\(dist=(\S+)\)$/g;if($2 eq "NONE" ){$gene=$3;}elsif( $4 eq "NONE"){$gene=$1;}elsif ($2<=$4){$gene=$1;print "$a[2]\t$a[3]\t$gene\t$a[0]\n"} else{  $gene= $3;print "$a[2]\t$a[3]\t$gene\t$a[0]\n"} }else { if($a[0]=~/;/){my@funs=split/;/,$a[0];$a[1]=~s/\(\S*?\)//g; my@genes=split/;/,$a[1]; for my$i(0..$#funs){ my@subgenes=split/,/,$genes[$i];for my$g(@subgenes){my$gg=$g;$gg=~s/\(\S+\)//g; print "$a[2]\t$a[3]\t$gg\t$funs[$i]\n";}  }}else{ $a[1]=~s/\(\S*?\)//g;  my@subgenes=split/,/,$a[1];for my$g(@subgenes){my$gg=$g;$gg=~s/\(\S+//g; print "$a[2]\t$a[3]\t$gg\t$a[0]\n";} } } '|sort -u|sed 's/\//_/g'|sort -k2n -k3 -k4 >chr$i.variant_function.uniqGene ;echo chr$i
python get_genes.py $i $file
