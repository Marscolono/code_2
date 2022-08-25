############################################################################################
#' @Example
#' This text provides an example of construction sample-specific driver modules 
#' (including sample-specific mutation driver module (ssMutat-DM), 
#' sample-specific methylation aberration driver module (ssMethy-DM) 
#' and sample-specific co-driver module (ssMethy-DM)) 
#' for each breast cancer sample.
############################################################################################

##########load expression data and background network
load(file="C:/Users/Dell/Desktop/manuscript.drive.net/投稿/code/data/normal.Rdata")   #reference samples
load(file="C:/Users/Dell/Desktop/manuscript.drive.net/投稿/code/data/cancer.Rdata")   #cancer samples
net <- read.table(file="C:/Users/Dell/Desktop/manuscript.drive.net/投稿/code/data/back.net.txt", header = T, sep="\t",stringsAsFactors = F) #background network
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
mutation <- read.table(file="C:/Users/Dell/Desktop/manuscript.drive.net/投稿/code/data/TCGA-BRCA.mutect2_snv.tsv", header = T, sep="\t", stringsAsFactors = F) #mutation data
mutation.matrix <- mutation2matrix(mutation) #mutation matrix

########Identify the aberrantly methylated genes for each sample, 
########and convert the hetylation data to a methylation aberrantion matrix
load(file="C:/Users/Dell/Desktop/manuscript.drive.net/投稿/code/data/metha.gene.Rdata") #methylation data
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


#####"TCGA.AR.A1AK.01A" "TCGA.E2.A15J.01A" "TCGA.AO.A1KS.01A" "TCGA.AC.A2BM.01A"
#####sample-specific subnetworks of driver modules for four breast cancer samples
mutat.s1  <- drive.net[[1]][[1]]
mutat.s1 <- mutat.s1[c(which(mutat.s1[, 1] == "PARP1"), which(mutat.s1[, 2] == "PARP1")), ]
write.table(mutat.s1, file="mutat.s1.txt", row.names = F, col.names = F, quote=F)

mutat.s2  <- drive.net[[1]][[2]]
mutat.s2 <- mutat.s2[c(which(mutat.s2[, 1] == "PARP1"), which(mutat.s2[, 2] == "PARP1")), ]
write.table(mutat.s2, file="mutat.s2.txt", row.names = F, col.names = F, quote=F)

mutat.s3  <- drive.net[[1]][[3]]
mutat.s3 <- mutat.s3[c(which(mutat.s3[, 1] == "PARP1"), which(mutat.s3[, 2] == "PARP1")), ]
write.table(mutat.s3, file="mutat.s3.txt", row.names = F, col.names = F, quote=F)

mutat.s4  <- drive.net[[1]][[4]]
mutat.s4 <- mutat.s4[c(which(mutat.s4[, 1] == "PARP1"), which(mutat.s4[, 2] == "PARP1")), ]
write.table(mutat.s4, file="mutat.s4.txt", row.names = F, col.names = F, quote=F)

intersect(intersect(paste(mutat.s1[, 1], mutat.s1[, 2], sep="$"), paste(mutat.s2[, 1], mutat.s2[, 2], sep="$")),
intersect(paste(mutat.s3[, 1], mutat.s3[, 2], sep="$"), paste(mutat.s4[, 1], mutat.s4[, 2], sep="$")))




mutat.s1  <- drive.net[[1]][[1]]
TP53.mutat.s1 <- mutat.s1[c(which(mutat.s1[, 1] == "TP53"), which(mutat.s1[, 2] == "TP53")), ]

mutat.s2  <- drive.net[[1]][[2]]
TP53.mutat.s2 <- mutat.s2[c(which(mutat.s2[, 1] == "TP53"), which(mutat.s2[, 2] == "TP53")), ]

mutat.s3  <- drive.net[[1]][[3]]
TP53.mutat.s3 <- mutat.s3[c(which(mutat.s3[, 1] == "TP53"), which(mutat.s3[, 2] == "TP53")), ]

mutat.s4  <- drive.net[[1]][[4]]
TP53.mutat.s4 <- mutat.s4[c(which(mutat.s4[, 1] == "TP53"), which(mutat.s4[, 2] == "TP53")), ]
intersect(intersect(paste(TP53.mutat.s1[, 1], TP53.mutat.s1[, 2], sep="$"), paste(TP53.mutat.s2[, 1], TP53.mutat.s2[, 2], sep="$")),
intersect(paste(TP53.mutat.s3[, 1], TP53.mutat.s3[, 2], sep="$"), paste(TP53.mutat.s4[, 1], TP53.mutat.s4[, 2], sep="$")))




methy.s1  <- drive.net[[2]][[1]]
methy.s1 <- methy.s1[c(which(methy.s1[, 1] == "TP53"), which(methy.s1[, 2] == "TP53")), ]
write.table(methy.s1, file="methy.s1.txt", row.names = F, col.names = F, quote=F)

methy.s2  <- drive.net[[2]][[2]]
methy.s2 <- methy.s2[c(which(methy.s2[, 1] == "TP53"), which(methy.s2[, 2] == "TP53")), ]
write.table(methy.s2, file="methy.s2.txt", row.names = F, col.names = F, quote=F)

methy.s3  <- drive.net[[2]][[3]]
methy.s3 <- methy.s3[c(which(methy.s3[, 1] == "TP53"), which(methy.s3[, 2] == "TP53")), ]
write.table(methy.s3, file="methy.s3.txt", row.names = F, col.names = F, quote=F)

methy.s4  <- drive.net[[2]][[4]]
methy.s4 <- methy.s4[c(which(methy.s4[, 1] == "TP53"), which(methy.s4[, 2] == "TP53")), ]
write.table(methy.s4, file="methy.s4.txt", row.names = F, col.names = F, quote=F)
intersect(intersect(paste(methy.s1[, 1], methy.s1[, 2], sep="$"), paste(methy.s2[, 1], methy.s2[, 2], sep="$")),
          intersect(paste(methy.s3[, 1], methy.s3[, 2], sep="$"), paste(methy.s4[, 1], methy.s4[, 2], sep="$")))




co.s1  <- drive.net[[3]][[1]]
co.s1 <- co.s1[c(which(co.s1[, 1] == "TP53"), which(co.s1[, 2] == "TP53")), ]
write.table(co.s1, file="co.s1.txt", row.names = F, col.names = F, quote=F)

co.s2  <- drive.net[[3]][[2]]
co.s2 <- co.s2[c(which(co.s2[, 1] == "TP53"), which(co.s2[, 2] == "TP53")), ]
write.table(co.s2, file="co.s2.txt", row.names = F, col.names = F, quote=F)

co.s3  <- drive.net[[3]][[3]]
co.s3 <- co.s3[c(which(co.s3[, 1] == "TP53"), which(co.s3[, 2] == "TP53")), ]
write.table(co.s3, file="co.s3.txt", row.names = F, col.names = F, quote=F)

co.s4  <- drive.net[[3]][[4]]
co.s4 <- co.s4[c(which(co.s4[, 1] == "TP53"), which(co.s4[, 2] == "TP53")), ]
write.table(co.s4, file="co.s4.txt", row.names = F, col.names = F, quote=F)
intersect(intersect(paste(co.s1[, 1], co.s1[, 2], sep="$"), paste(co.s2[, 1], co.s2[, 2], sep="$")),
          intersect(paste(co.s3[, 1], co.s3[, 2], sep="$"), paste(co.s4[, 1], co.s4[, 2], sep="$")))





