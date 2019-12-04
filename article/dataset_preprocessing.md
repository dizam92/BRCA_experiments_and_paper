# Materiels et methodes section
----------
# Datasets
---------
1. Retrieve the datasets for each file view:
	1. methyl_27.tsv: (311, 27578) 
	2. methyl_450.tsv: (778, 485577)
	3. rna.tsv: (1083, 20531)
	4. rna_isoforms.tsv: (1083, 73599)
	5. mirna.tsv: (1067, 1046)
	6. clinical.tsv: (20, 934)
	7. methyl_fusion: (clinical_file_x, 23381)
	8. snps.tsv: (969, 4192)

2. https://www.biorxiv.org/content/biorxiv/early/2016/08/03/067611.full.pdf page 28 __In addition, the 3000 most variable genes based on the median absolute deviation (MAD) were retained for downstream analysis"__
	1. samples coming from the primary solid tumor (sample type code 01) and to the first vial (vial code A)
	2. Remove all features with nan
	3. Compute the median absolute deviation (MAD) per features (view) and select a random number fixed: 3000/2000....
		1. A propros du MAD: They didn't specified how the done it. Mais prudo me propose ceci: __pour chaque gene tu calcule le mad, ensuite tu sort les gene par mad et tu prends les 3000 ou n premiers__ Ceci comme l'article nous permetrait de vraiment selectionner les genes(features) avec le plus de grande variabilités donc susceptibles de mieux nous guidés? Dnas l'article ils ont considérer n == 2000 pour cpg et rna. Et beaucoup moins pour les mirnas.
		2. https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/ propose de faire des supposition sur la distribution et de chercher les outliers en faisant (x_value > coeff * mad) i.e prendre les x pour lesquelles les valeurs sont strictement superieur à ce calcul. Mais ca ca elimine les outliers (example) mais pas les features right. A moins que je fasse x.T mais essentiellement ca devrait marcher sur les examples? 

Summary Table

|View files| Original Shape| Shape after deleted the Nan features|
| ------------- |:-------------:| -----:|
|methyl_27.tsv| (311, 27578) | (311, 22533) |
|methyl_450.tsv| (778, 485577)| (778, 364520)|
|rna.tsv| (1083, 20531)| (1083, 20502)|
|rna_isoforms.tsv| (1083, 73599)| (1083, 73599) __probably no need to do something__|
|mirna.tsv| (1067, 1046)| (1067, 1046) __probably no need to do something__|
|clinical.tsv| (934, 20)| (934, 20 )|
|methyl_fusion| (clinical_file_x, 23381)| (clinical_file_x, 19984)|
|snps.tsv| (969, 4192) | (969, 4192) |


3. I found an error in the data while loading the view with snp and clinical. This does not impact the pretty good results i have so far just on the other 3 view so don't panick and delete everything. 
	1. We gon save the results we have so far here /home/maoss2/project/maoss2
	2. We rebuild the new datasets (histoire d'etre têteux): 
	3. Et on relance les expes froms scratch (should not be a nan probleme after that normalement)

