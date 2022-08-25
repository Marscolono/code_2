
ssMutat(Methy)-Driver Modules


-------------------------References----------------------------------------
# Here is the source code of the construction of samples-specific mutation driver modules and methylation aberration driver module （ssMutat-DM and ssMethy-DM).

References:
Yuanyuan Chen, Haitao Li, Xiao Sun, Construction and analysis of sample-specific driver modules for breast cancer.


--------------------------Description---------------------------------------------
The functional impact of somatic mutation and methylation aberration at an individual level is an important research direction to implement precision medicine. 
More and more studies have shown that the perturbation of gene interaction network provides a fundamental link between genotype (or epigenotype) and phenotype. 
However, functional effects on biological networks for individual mutations, especially for individual methylation aberration, are largely unknown. 
To solve this, we provided a sample-specific driver module construction method by the 2-order network theory and hub-gene theory 
to identify individual perturbation networks driven by mutations or methylation aberrations. 
The method integrated multi-omics including genomics, transcriptomics, epigenomics and interactomic of breast cancer, 
and provided new insight into the synergistic collaboration between methylation and mutation at individual level. 

We construct sample-specific driver modules in three steps. 
First, the sample-specific network (SSN) for each breast cancer sample can be constructed based on the gene expression profiles and background network by the SSN method. 
Second, the somatic mutation profiles of all breast cancer samples can be constructed by the somatic mutation data. 
In addition, based on the methylation data we can obtain the aberrantly methylated genes for each tumor sample by the outlier detection method of Hampel filter, 
which make up the methylation aberration profiles for breast cancer samples. 
Finally, the sample-specific mutation driver module (ssMutat-DM) and the sample-specific methylation aberration driver module (ssMethy-DM) 
can be constructed using the 2-order network theory and hub-gene theory.



Example
############################################################################################
#' @Example
#' This text provides an example of construction sample-specific driver modules 
#' (including sample-specific mutation driver module (ssMutat-DM), 
#' sample-specific methylation aberration driver module (ssMethy-DM) 
#' and sample-specific co-driver module (ssMethy-DM)) 
#' for each breast cancer sample.
############################################################################################

##########load expression data and background network
load(file="....../data/normal.Rdata")   #reference samples
load(file="....../data/cancer.Rdata")   #cancer samples
net <- read.table(file="....../data/back.net.txt", header = T, sep="\t",stringsAsFactors = F) #background network
net <- data.frame(N1=net[,1], N2=net[,2])
net <- net[!duplicated(net),]
net <- igraph::graph_from_data_frame(net, directed = F, vertices = NULL)
net <- igraph::simplify(net, remove.loops = T)
net <- as.data.frame(igraph::get.edgelist(net))

########Construct sample-specific network(SSN) for 10 random selected breast cancer samples
n <- 10  #Number of randomly selected samples
set.seed(123445)
cancer <- cancer[, sample(1:1097, n)]
SSN <- PCC(normal, cancer, net, n.kernal= 2)
save(SSN, file="SSN.Rdata")

#########convert the mutation data to a mutation matrix
mutation <- read.table(file="....../data/TCGA-BRCA.mutect2_snv.tsv", header = T, sep="\t", stringsAsFactors = F) #mutation data
mutation.matrix <- mutation2matrix(mutation) #mutation matrix

########Identify the aberrantly methylated genes for each sample, 
########and convert the hetylation data to a methylation aberrantion matrix
load(file="....../data/metha.gene.Rdata") #methylation data
DMG.matrix <- DMG(metha.gene) #methylation aberrantion matrix 


###############construct the ssMutat-DM, ssMethy-DM and ssCo-DM for each patient 
library(igraph)
drive.net <- ss.drive.net(SSN, mutation.matrix, DMG.matrix, net)


#############example of Drive network visualization for sample "TCGA.E2.A15J.01A"
############or you can export the network and draw pictures in the cytoscape
ssMutat.DM  <- drive.net[[1]][[1]]
write.table(ssMutat.DM, file="ssMutat.DM.txt", row.names = F, col.names = F, quote=F)
edges <- c(t(ssMutat.DM))
ssMutat.DM.net <- graph(edges, n = ncol(ssMutat.DM), directed = F)
plot(ssMutat.DM.net)

ssMethy.DM  <- drive.net[[2]][[1]]
write.table(ssMethy.DM, file="ssMethy.DM.txt", row.names = F, col.names = F, quote=F)
edges <- c(t(ssMethy.DM))
ssMethy.DM.net <- graph(edges, n = ncol(ssMethy.DM), directed = F)
plot(ssMethy.DM.net)


ssco.DM  <- drive.net[[3]][[1]]
write.table(ssco.DM, file="ssco.DM.txt", row.names = F, col.names = F, quote=F)
edges <- c(t(ssco.DM))
ssco.DM.net <- graph(edges, n = ncol(ssco.DM), directed = F)
plot(ssco.DM.net)




--------------------------Contact---------------------------------------------
If any problem or suggestion, please contact Meina Kan (chenyuanyuan@njau.edu.cn)
