############################################################################################
#' @description Construct sample-specific network(SSN) for each breast cancer sample based on expression matrix and gene interaction network.

#' @param normal, The expression matrix of normal samples.
#' @param cancer, The exprssion matirx of cancer samples.
#' @param net, The background network in the format of matrix with two colums which repsents edges.
#' @n, The number of threads running in parallel.

#' @return @perturb.matrix, row:edges, col:samples. The sample-specific network: 0-1 matrix,1 represents significant edges, otherwise 0.
#'  the all "1"-edges of a single sample constitute this sample-specific network
############################################################################################

PCC <- function(normal, cancer, net, n.kernal){
  pair <-list() #Store the expression data of each gene pair
  oldpcc <- c()  #caculate the PCCn by normal samples
  for(i in 1:nrow(net)){
    pair[[i]] <- subset(normal, row.names(normal) %in% net[i, 1] | row.names(normal) %in% net[i, 2])
    y <- t(pair[[i]])
    y <- apply(y, 2, as.numeric)
    oldpcc[i] <- cor(y, method = 'pearson')[1,2]
  }
  oldpcc <<- as.numeric(oldpcc)
  
  
  #Add each patient sample to the normal sample respectively
  del <- function(j){
    normws_j <- cbind(normal, cancer[,j]) #Add the jth cancer sample to the normal samples
    pcc <- c()
    for(i in 1:nrow(net)){
      pair <- subset(normws_j, row.names(normws_j) %in% net[i, 1] | row.names(normws_j) %in% net[i, 2])
      y <- apply(t(pair), 2, as.numeric)
      pcc <- rbind(pcc, cor(y, method = 'pearson')[1, 2])
    }
    #PCCn+1
    deltapcc <- pcc[, 1]- oldpcc #deltaPCC=PCCn+1-PCCn
  }
  #matrix M--each column represents the deltapcc of all gene pairs beacuse of a single cancer sample
  
  library(parallel)
  cl <- makeCluster(n.kernal)
  clusterExport(cl, c('normal', 'cancer', "net", "oldpcc"), envir = .GlobalEnv)
  Deltapcc <- parLapply(cl, 1 : ncol(cancer), del)
  Deltapcc.matrix<- data.frame(matrix(unlist(Deltapcc), nrow=length(Deltapcc[[1]]), byrow=F), stringsAsFactors=FALSE)
  stopCluster(cl)
  
  #Z-test
  ssn_score <- function(deltapcc, pcc, nn){
    if(pcc == 1){
      pcc=0.999999
    }
    if(pcc == -1){
      pcc=-0.99999
    }
    z <- deltapcc/((1-pcc*pcc)/(nn-1))
  }
  
  ##Find the significant gene pairs for each patient through Z-test
  z <- apply(Deltapcc.matrix, 2, ssn_score, nn=113, pcc=oldpcc)
  z_net <- function(zt){
    x <- runif(length(zt), min=0, max=0)
    pvalue <- 1 - pnorm(abs(zt))
    x[which(pvalue<0.01)] <- 1
    return(x)
  }
  perturb.net <- apply(z, 2, z_net)
  perturb.net <- as.data.frame(perturb.net) 
  perturb.net <- apply(perturb.net, 2, as.numeric)
  colnames(perturb.net) <-  colnames(cancer) 
  return(perturb.net)
}
############################################################################################


########################################################################################################
#' @description, convert the mutation data to a mutation matrix
#' @param mutation, mutation data
#' @return @mutat.matrix, mutation matrix, row:genes, col:samples. 0-1 matrix.
#########################################################
mutation2matrix <- function(mutation){
  mutation.genes <- unique(mutation$gene)
  mutat.matrix <- matrix(0, nrow= length(mutation.genes), ncol= length(table(mutation$Sample_ID)))
  row.names(mutat.matrix) <- names(table(mutation$gene))
  colnames(mutat.matrix) <- names(table(mutation$Sample_ID))
  for (i in 1:nrow(mutation)){
    mutat.matrix[mutation$gene[i], mutation$Sample_ID[i]] <- 1
  }
  return(mutat.matrix)
}
############################################################################



#####################################################################################
# Description, Identify the aberrantly methylated genes for each sample by the outlier detection method of Hampel filter.
#' @metha.gene, matrix, the methylation value in gene level.
#' @return  @DMG.matrix, methylation aberrantion matrix, row:genes, colum:samples.
###############################################################
DMG <- function(metha.gene){   #hampel, median+- 3*mad
  metha.gene <- na.omit(metha.gene) 
  metha.normal <- metha.gene[, which(substr(colnames(metha.gene), 14, 15) =="11")]
  metha.cancer <- metha.gene[, which(substr(colnames(metha.gene), 14, 15) !="11")]
  DMG.matrix <- matrix(0, nr = nrow(metha.normal), ncol = ncol(metha.normal)+ncol(metha.cancer))
  for (i in 1:nrow(metha.normal)){
    if(i%%1000 == 0) {print(i)}
    d <- c(metha.normal[i, ], metha.cancer[i,])
    DMG.matrix[i, which(d > median(d) + 3*mad(d))] <- 1
    DMG.matrix[i, which(d < median(d) - 3*mad(d))] <- -1
  }
  colnames(DMG.matrix) <- c(colnames(metha.normal), colnames(metha.cancer))
  row.names(DMG.matrix) <- row.names(metha.normal)
  return(DMG.matrix)
}
###########################################################################




###############################################################################################
#description, construct the ssMutat-DM and ssMethy-DM for each patient based on the 2-order network theory and hub-gene theory.
#' @perturb.matrix, perturb matrix by sample-specific network method
#' @mutat.matrix, mutation matrix
#' @DMG.matrix, methylation aberrantion matrix
#' @net, background network

#' @return
#' @mutat.drive.nets, the sample-specific mutation driver matrix(ssMutat-DM) 
#' @methy.drive.nets, the sample-specific methylation driver matrix(ssMethy-DM)
#' @co.drive.nets, the sample-specific co-driver matrix(ssCo-DM)
#############################################################

ss.drive.net <- function(perturb.matrix, mutat.matrix, DMG.matrix, net){
    colnames(perturb.matrix) <- gsub("\\_", "\\.", colnames(perturb.matrix)) 
    colnames(mutat.matrix) <- gsub("\\-", "\\.", colnames(mutat.matrix)) 
    samples <- intersect(intersect(colnames(DMG.matrix), colnames(perturb.matrix)), colnames(mutat.matrix))
    methy <- DMG.matrix[, samples]
    mutat <- mutat.matrix[, samples]
    perturb.edges <- perturb.matrix [, samples]
  
    sort(apply(perturb.edges, 1, mean), decreasing = T)[1:50]
    net[order(apply(perturb.edges, 1, mean), decreasing = T)[1:100], ]
  
 #identify drive nets 
    mutat.drive.nets <- list()
    methy.drive.nets <- list()
    library(igraph)
    for (i in 1:length(samples)){
       mutat.gene <- row.names(mutat)[which(mutat[,i]==1)]
       methy.gene <- row.names(methy)[c(which(methy[,i]==1), which(methy[,i]==-1))]
      if(length(mutat.gene) > 20){
          g <- net[which(perturb.edges[, i]==1),]
          g <- graph.data.frame(g, directed = F)
          deg <- degree(g)
          deg <- deg[order(degree(g), decreasing = T) ]
          hub.genes <- names(deg[1:round(0.2*length(deg))])# the first 20% genes with more degree
          mutat.subnet <- make_ego_graph(g, order=2, intersect(mutat.gene, names(deg))) 
      #subnetwork construced by the 2-order neighborhood vertex of mutation genes in the sample-specific perturb network
          methy.subnet <- make_ego_graph(g, order=2, intersect(methy.gene, names(deg)))
          mutat.net <- mutat.subnet[[1]]
          for (j in 1:length(mutat.subnet)){
              mutat.net <-union(mutat.net, mutat.subnet[[j]])
          }
          mutat.net <- subgraph(graph = mutat.net, v = c(intersect(mutat.gene, names(degree(mutat.net))), intersect(hub.genes, names(degree(mutat.net)))))
      #the subnetwork with mutation genes and all other genes are the hub-genes in the 2-order subnetwork
          mutat.drive.net <- as_data_frame(mutat.net)
          mutat.drive.nets[[i]] <- mutat.drive.net  
      
          methy.net <- methy.subnet[[1]]
         for (j in 1:length(methy.subnet)){
               methy.net <-union(methy.net, methy.subnet[[j]])
         }
         methy.net <- subgraph(graph = methy.net, v = c(intersect(methy.gene, names(degree(methy.net))), intersect(hub.genes, names(degree(methy.net)))))
         methy.drive.net <- as_data_frame(methy.net)
         methy.drive.nets[[i]] <- methy.drive.net 
    }
    #drive.genes <- intersect(mutat.gene, hub.genes)
    #drive.genes <- names(degree(mutat.net))[order(degree(mutat.net), decreasing = T)][1:20]
    #drive.genes <- intersect(mutat.gene, drive.genes)
  }
  co.drive.nets <- list()
  for(i in 1:length(samples)){
     data <- rbind(mutat.drive.nets[[i]], methy.drive.nets[[i]])
     data <- data[!duplicated(data), ]
     co.drive.nets[[i]] <- data
  }
  return(list(mutat.drive.nets, methy.drive.nets, co.drive.nets))
}

