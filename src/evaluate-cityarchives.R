library(data.table)
library(ggplot2)
library(ggfortify)
library(Metrics)
library(pbapply)
library(caret)
library(kernlab)
library(Rcpp)
library(RcppArmadillo)
library(RcppProgress)
library(Matrix)
library(Matrix.utils)
sourceCpp('src/functions.cpp')
sourceCpp('src/cluster-editing.cpp')
source('src/functions.R')


##################
#    Clusters    #
##################

# Load in ground truth cluster data
clusters <- fread('data/saa_clusters.tsv', header = T, sep = '\t', blank.lines.skip = T)
clusters <- clusters[!is.na(clusters$ILN_GROUP)]
clusters$AL_ID <- as.integer(as.factor(clusters$CLUSTER_ID))
clusters$CLUSTER_ID <- as.integer(as.factor(clusters$CLUSTER_ID))
clusters$CLUSTER_ID <- as.integer(paste0(clusters$CLUSTER_ID, clusters$ILN_GROUP))
clusters$RESOURCE_URI <- paste0(clusters$DATASET, clusters$RESOURCE_URI)
# Clear all unused columns
clusters$STATUS <- NULL
clusters$CLUSTER_COUNT <- NULL
clusters$DATASET <- NULL
clusters$CLUSTER_SIZE <- NULL
clusters$ILN_GROUP <- NULL


################
#    pGloVe    #
################

# Load in the data
filename <- createFileName('saa', 'pglove', T, 'directed_weighted', 'amsgrad', 500)

vectors.pglove <- fread(paste0('../graph-embeddings/out/', filename, '.vectors.tsv'), sep = "\t")
keys <- fread(paste0('../graph-embeddings/out/',filename,'.dict.tsv'), sep = '\t', quote = "")

# Only keep the records for URI's
uris <- keys$type == 0
vectors.pglove <- vectors.pglove[uris]
keys <- keys[uris]$key

# Only keep the records for persons
persons <- grepl('person', keys) 
keys <- keys[persons]
vectors.pglove <- vectors.pglove[persons]
colnames(vectors.pglove) <-  paste0('d', 1:ncol(vectors.pglove))

clusters <- merge(data.table(RESOURCE_URI = keys, ID = 1:length(keys)), clusters, all = T)
setorder(clusters, ID)
clusters$CLUSTER_ID[is.na(clusters$CLUSTER_ID)] <- -1
clusters$AL_ID[is.na(clusters$AL_ID)] <- -1
clusters$ID <- NULL

#vectors.pglove.dist <- data.table(Rfast::Dist(vectors.pglove, method = distanceMetric))
tic('R')
ndcg.pglove <- ndcg(vectors.pglove, clusters$CLUSTER_ID)
toc()
tic('c++')
ndcg.pglove <- m2v(calculateNDCG(as.matrix(vectors.pglove), clusters$CLUSTER_ID))
toc()

mean.cluster.size <- mean(clusters[CLUSTER_ID != -1, .N, by=CLUSTER_ID]$N)

dataset.pglove <- generate.train.set(vectors.pglove, clusters$CLUSTER_ID, ceiling(mean.cluster.size))
dataset.pglove <- as.data.table(dataset.pglove)
dataset.pglove$y <- as.factor(dataset.pglove$y)
# Caret does not like class labels of '0' and '1'
levels(dataset.pglove$y) <- c('no','yes')

hyper.sample.pglove <- createDataPartition(dataset.pglove$y, p = 0.2, list = F)

fitControl <- trainControl (
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "random",
  verboseIter = TRUE
)

svm.pglove.tune <- train (
  y ~ ., 
  data = dataset.pglove[hyper.sample.pglove], 
  method = "svmRadial",
  metric = "ROC",
  trControl = fitControl,
  preProc = c('scale', 'center'),
  tuneLength = 30
)

train.sample.pglove <- createDataPartition(dataset.pglove[-hyper.sample.pglove]$y, p = 0.75, list = F)

svm.pglove.final <- kernlab::ksvm (
  y ~ .,
  data = dataset.pglove[-hyper.sample.pglove][train.sample.pglove],
  kernel = 'rbfdot',
  C = svm.pglove.tune$bestTune$C,
  kpar = list(sigma = svm.pglove.tune$bestTune$sigma),
  prob.model = T
)

svm.results.pglove <- predict(svm.pglove.final, dataset.pglove[-hyper.sample.pglove][-train.sample.pglove, -501], type = 'probabilities')
svm.results.pglove <- data.table(actual = as.integer(dataset.pglove[-hyper.sample.pglove][-train.sample.pglove]$y) - 1, prediction = as.integer(svm.results.pglove) - 1)
table(svm.results.pglove)

pglove.accuracy <- Metrics::accuracy(svm.results.pglove$actual, svm.results.pglove$prediction)
pglove.precision <- Metrics::precision(svm.results.pglove$actual, svm.results.pglove$prediction)
pglove.recall <- Metrics::recall(svm.results.pglove$actual, svm.results.pglove$prediction)
pglove.fbeta <- Metrics::fbeta_score(svm.results.pglove$actual, svm.results.pglove$prediction, beta = 1)

library(tictoc)
tic();
clusters$JURIAN_ID <- findEquivalenceClasses(
  createRelationalMatrixFast(t(as.matrix(vectors.pglove)), 10, 10, svm.pglove.final, kernlab::predict)
)
toc();

clusters[, N := .N, by = .(JURIAN_ID_2)]
clusters$JURIAN_ID_2[clusters$N == 1] <- -1
clusters$N <- NULL

fwrite(clusters[CLUSTER_ID != -1 | AL_ID != -1 | JURIAN_ID != -1], file = 'SAA_clusters.csv', col.names = T, row.names = F)

rm(vectors.pglove)

#sum(svm.results$actual == svm.results$prediction) / nrow(svm.results)


#distances <- generate.train.set(vectors.pglove.dist, 25)

#plot.distances(sample(which(clusters$ndcg == 1), 12), 20, clusters)
#plot.distances(sample(which(clusters$ndcg < 0.5), 12), 20, clusters)  
#plot.densities(sample(which(clusters$ndcg == 1), 12), 20, clusters)
#plot.densities(sample(which(clusters$ndcg < 0.5), 12), 20, clusters)

################
#     GloVe    #
################

filename <- createFileName('saa', 'glove', T, 'directed_weighted', 'amsgrad', 500)

# Load in the data
vectors.glove <- fread(paste0('../graph-embeddings/out/', filename, '.vectors.tsv'), sep = "\t")
vectors.glove <- vectors.glove[uris]
vectors.glove <- vectors.glove[persons]
colnames(vectors.glove) <-  paste0('d', 1:ncol(vectors.glove))

# Clean up
rm(uris, keys, persons)

#vectors.glove.dist <- data.table(Rfast::Dist(vectors.glove, method = distanceMetric))
ndcg.glove <- ndcg(vectors.glove, clusters$CLUSTER_ID)

dataset.glove <- generate.train.set(vectors.glove, clusters$CLUSTER_ID, ceiling(mean.cluster.size))
dataset.glove <- as.data.table(dataset.glove)
dataset.glove$y <- as.factor(dataset.glove$y)
# Caret does not like class labels of '0' and '1'
levels(dataset.glove$y) <- c('no','yes')



hyper.sample.glove <- createDataPartition(dataset.glove$y, p = 0.2, list = F)

svm.glove.tune <-  train (
  y ~ ., 
  data = dataset.glove[hyper.sample.glove], 
  method = "svmRadial",
  metric = "ROC",
  trControl = fitControl,
  preProc = c('scale', 'center'),
  tuneLength = 10
)

# Remove hyper parameter sample
train.sample.glove <- createDataPartition(dataset.glove[-hyper.sample.glove]$y, p = 0.75, list = F)

svm.glove.final <- ksvm (
  y ~ .,
  data = dataset.glove[-hyper.sample.glove][train.sample.glove],
  kernel = 'rbfdot',
  C = svm.glove.tune$bestTune$C,
  kpar = list(sigma = svm.glove.tune$bestTune$sigma),
  prob.model = T
)

svm.results.glove <- predict(svm.glove.final, dataset.glove[-hyper.sample.glove][-train.sample.glove, -501])
svm.results.glove <- data.table(actual = as.integer(dataset.glove[-hyper.sample.glove][-train.sample.glove]$y) - 1, prediction = as.integer(svm.results.glove) - 1)
table(svm.results.glove)

glove.accuracy <- Metrics::accuracy(svm.results.glove$actual, svm.results.glove$prediction)
glove.precision <- Metrics::precision(svm.results.glove$actual, svm.results.glove$prediction)
glove.recall <- Metrics::recall(svm.results.glove$actual, svm.results.glove$prediction)
glove.fbeta <- Metrics::fbeta_score(svm.results.glove$actual, svm.results.glove$prediction, beta = 1)


clusters$EQCLASS_GLOVE <- findEquivalenceClasses(
  generate.relational.matrix(vectors.glove, svm.glove.final, 100)
)
rm(vectors.glove)


plot.dt <- rbind(data.table(ndcg = ndcg.glove, type = 'glove'), data.table(ndcg = ndcg.pglove, type = 'pglove'))

ggplot(plot.dt,aes(x=ndcg)) + 
  geom_histogram(data=subset(plot.dt,type=='glove'),aes(fill=type), bins = 200) +
  geom_histogram(data=subset(plot.dt,type=='pglove'),aes(fill=type), bins = 200) +
  scale_fill_manual(name="type", values=c('firebrick3','dodgerblue3'),labels=c("GloVe","pGloVe")) +
  xlab('Normalized Discounted Cumulative Gain') + 
  ylab('Frequency') +
  theme_classic() +
  theme(axis.title=element_text(size=14), axis.text=element_text(size=16), legend.text = element_text(size=16), legend.title = element_blank())

sum(ndcg.pglove == 1 ) / length(ndcg.pglove)
sum(ndcg.glove == 1 ) / length(ndcg.glove)


temp <- dataset.pglove[sample(nrow(dataset.pglove), 5000)]
autoplot(prcomp(temp[,-501], scale = T), data = temp, colour = 'y') + 
  scale_color_manual(name="Dulicate", values=c('dodgerblue3','firebrick3'),labels=c("No","Yes")) +
  coord_cartesian(ylim = c(-0.05, 0.05)) +
  theme_classic() +
  theme(axis.title=element_text(size=14), axis.text=element_text(size=16), legend.text = element_text(size=16))

temp2 <- dataset.glove[sample(nrow(dataset.glove), 5000)]
autoplot(prcomp(temp2[,-501], scale = T), data = temp2, colour = 'y') + 
  scale_color_manual(name="Dulicate", values=c('dodgerblue3','firebrick3'),labels=c("No","Yes")) +
  coord_cartesian(xlim = c(-0.025, 0.05), ylim = c(-0.05, 0.05)) +
  theme_classic() +
  theme(axis.title=element_text(size=14), axis.text=element_text(size=16), legend.text = element_text(size=16))

