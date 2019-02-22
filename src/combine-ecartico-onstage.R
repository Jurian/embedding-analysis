library(data.table)
library(ggplot2)
library(Rtsne)

# Load in the data

linkset <- fread('data/ecartico.onstage.linkset.csv', header = F)
colnames(linkset) <- c('onstage', 'ecartico')

ecartico.vectors <- fread('data/ecartico.nt.amsgrad.50.vectors.txt')
ecartico.keys <- fread('data/ecartico.nt.amsgrad.50.dict.txt')

onstage.vectors <- fread('data/onstage.nt.amsgrad.50.vectors.txt')
onstage.keys <- fread('data/onstage.nt.amsgrad.50.dict.txt')


uris <- grepl('www.vondel.humanities.uva.nl/ecartico', ecartico.keys$V1)
ecartico.vectors <- ecartico.vectors[uris]
ecartico.keys <- ecartico.keys[uris]$V1

uris <- grepl('www.vondel.humanities.uva.nl/onstage', onstage.keys$V1)
onstage.vectors <- onstage.vectors[uris]
onstage.keys <- onstage.keys[uris]$V1

linkset$onstage <- sapply(linkset$onstage, function(x){
   which(onstage.keys %in% x)
})
linkset$ecartico <- sapply(linkset$ecartico, function(x){
  which(ecartico.keys %in% x)
})

combined.data <- data.frame(onstage = onstage.vectors[linkset$onstage], ecartico = ecartico.vectors[linkset$ecartico])
