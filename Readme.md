# BRCA STUFF

# PRAD STUFF

# Informations fournies par Steve

1. Metastatic Prostate Adenocarcinoma (SU2C/PCF Dream Team, PNAS 2019)__(444 total samples) - 444 métastases__
2. Prostate Adenocarcinoma (Broad/Cornell, Nat Genet 2012)__(112 total samples) - 112 tumeurs primaires__
3. Prostate Adenocarcinoma (Fred Hutchinson CRC, Nat Med 2016)__(176 total samples) - 154 métastases et 22 tumeurs primaires__
4. Prostate Adenocarcinoma (SMMU, Eur Urol 2017) - __65 tumeurs primaires__ (RNA-seq pas sur le cBioPortal par contre) __IGNORER__
5. Prostate Adenocarcinoma (TCGA, Firehose Legacy)__(499 total samples) - 499 tumeurs primaires__

# Recapitulatif
1. Nbre de __tumeurs primaires__: 499 + 22 + 112  = __633__ (-1)
2. Nbre de __metastases__: __444__ (1)

# Informations disponibles
1. On exclue les fichiers cliniques vu qu'ils ne sont pas uniformes d'études en études
2. __mRNA__ exclusivement à considérer
3. __CNA__ (Copy Number Alteration) aussi disponible 
4. SNPS (on peut reconstruire les snps) mais explosions de dimensions? Est-ce nécessaire ici? 
5. TCGA a les données methylome disponible et RPPA (sur les protéines aussi) (Idées d'entrainer avec des données sparse?)

# Apres traitements
1. (8619, 332) metastases examples (cna and mRNA)
2. 

# DBGAP INFORMATIONS To ACCESS AND DOWNLOAD
1. Download and install the sra tools: https://github.com/ncbi/sra-tools/wiki/02.-Installing-SRA-Toolkit
2. Add it into your .basrc (optional but pratical to use the command line directly)
3. Get the dbGap repository key (from the dbgap project): I suggest you to download directly on your computer then scp it on the server (here i am using graham) name of the file: prj_8983.ngc
4. On the dbgap site:
    1. create the cart list of what you want to download (cart list is easier) or go with the accession list too (didn't try it though)
    2. prefeth --ngs <path to prj_####.ngc> --cart <path to cart_prj####_###.krt> exemple: prefetch --ngc prj_8983.ngc --cart cart_prj8983_202007161420.krt
    3. Override the maximum size of the dataset to download: 
    prefetch --ngc prj_8983.ngc --max-size 100000000 --cart cart_prj8983_202007161420.krt --progress --output-directory /home/maoss2/project/maoss2/dbGap_database

5. [https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?view=toolkit_doc&f=dbgap_use]
