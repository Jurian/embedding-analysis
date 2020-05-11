
library(data.table)
library(kernlab)
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
library(pbapply)
library(Matrix)

rm(list=lsf.str())
Rcpp::sourceCpp('src/CppFunctions.cpp')

###############################################################
#                     DEFINE FUNCTIONS                        #
###############################################################

get.ground.truth <- function() {
  # Get ground truth data
  truth <- fread('data/saa/saa.ground.truth.tsv', header = T, sep = '\t', blank.lines.skip = T)
  truth <- truth[!is.na(truth$ILN_GROUP)]
  truth$key <- paste0(truth$DATASET, truth$RESOURCE_URI)
  truth$clstID_AL <- as.integer(as.factor(truth$CLUSTER_ID))
  truth$clstID <- as.integer(as.factor(truth$CLUSTER_ID))
  truth$clstID <- as.integer(paste0(truth$clstID, truth$ILN_GROUP))
  truth$index <- 1:nrow(truth)
  
  # Clear all unused columns
  truth$STATUS <- NULL
  truth$CLUSTER_COUNT <- NULL
  truth$DATASET <- NULL
  truth$CLUSTER_SIZE <- NULL
  truth$ILN_GROUP <- NULL
  truth$CLUSTER_ID <- NULL
  truth$RESOURCE_URI <- NULL
  
  return(truth)
}

ground.truth.metrics <- function(partition) {
  
  truth <- get.ground.truth()
  
  # Search within Al's clusters for pairs and label them positive or negative according to subclusters (if any)
  grid.inner <- truth[, data.table(t(combn(x = index, m = 2))), by = .(clstID_AL)]
  grid.inner[, valid := as.integer(truth[V1]$clstID == truth[V2]$clstID)] # A combination is valid if they have the same cluster ID
  grid.inner[, valid_AL := as.integer(truth[V1]$clstID_AL == truth[V2]$clstID_AL)]
  
  missing <- abs(sum(grid.inner$valid == 0) - sum(grid.inner$valid == 1))
  
  grid.inner[, clstID_AL := NULL]
  grid.inner[, V1 := NULL]
  grid.inner[, V2 := NULL]
  
  grid.inner <- rbind(grid.inner, data.table(valid = rep(0, missing), valid_AL = rep(0, missing)))
  
  grid.inner[, valid := as.factor(valid)]
  grid.inner[, valid_AL := as.factor(valid_AL)]
  
  levels(grid.inner$valid) <- c('no', 'yes')
  levels(grid.inner$valid_AL) <- c('no', 'yes')
  
  grid.inner[, valid := relevel(valid, 'yes')]
  grid.inner[, valid_AL := relevel(valid_AL, 'yes')]
  
  caret::confusionMatrix(grid.inner$valid_AL[partition], grid.inner$valid[partition])
}

# Generate global pairs of vectors based on the ground truth
get.key.pairs <- function() {
  
  # Get ground truth data
  truth <- get.ground.truth()
  
  # Search within Al's clusters for pairs and label them positive or negative according to subclusters (if any)
  grid.inner <- truth[, data.table(t(combn(x = index, m = 2))), by = .(clstID_AL)]
  grid.inner[, valid := as.integer(truth[V1]$clstID == truth[V2]$clstID)] # A combination is valid if they have the same cluster ID
  grid.inner$clstID_AL <- NULL # Remove variable we grouped on
  
  # Search between Al's clusters for pairs and label them negative
  grid.outer <- data.table(t(combn(x = truth$index, m = 2))) # Get all combinations
  grid.outer <- grid.outer[truth[V1]$clstID_AL != truth[V2]$clstID_AL] # Remove inner combinations
  grid.outer[, valid := 0] # All outer combinations are invalid
  
  # Sample enough negative sample to make the positive-negative pairs ratio 50-50
  negative.sample <- sample(nrow(grid.outer), abs(sum(grid.inner$valid == 0) - sum(grid.inner$valid == 1)))
  
  # Combine inner and part of outer grid
  grid.combn <- rbind(grid.inner, grid.outer[negative.sample])
  
  colnames(grid.combn) <- c('key1', 'key2', 'y')
  
  # Use the URI's as keys instead of indexes
  grid.combn[, key1 := truth[key1]$key]
  grid.combn[, key2 := truth[key2]$key]
  
  # Caret doesn't like it when you do classification and y is not a factor
  grid.combn[, y := as.factor(y)]
  levels(grid.combn$y) <- c('negative', 'positive')
  grid.combn[, y := relevel(y, 'positive')]
  
  return(grid.combn)
}

# Partition the pairs into a train and test set, while keeping ratios intact
partition.pairs <- function(pairs, p = 0.5, sample = NULL) {
  
  if(is.null(sample))
    partition <- caret::createDataPartition(y = pairs$y, p = p, list = F)
  else
    partition <- sample
  
  pairs.split <- list()
  
  pairs.split$train <- pairs[partition]
  pairs.split$test <- pairs[-partition]
  
  return(pairs.split)
}

# Calculate hadmard products of global pairs using a given embedding
hadamard <- function(edges, matching, key.pairs) {
  
  # Get the data based on matching type and edge directioness
  vect <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.vectors.tsv'), header = F, sep = "\t")
  dict <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.dict.tsv'), header = T, sep = "\t")
  dict$index <- 1:nrow(dict)
  dict$type <- NULL
  
  # Vect and dict are in the same order, but key pairs is not!
  # Using the keys in dict, merge with the keys in the global key pairs
  index <- data.table(
    key1 = merge(key.pairs, dict, by.x = 'key1', by.y = 'key', sort = F)$index,
    key2 = merge(key.pairs, dict, by.x = 'key2', by.y = 'key', sort = F)$index
  )
  
  # Now we know the index in vect for each global key pair
  # Proceed to calculate the hadmard product for each pair
  hadamard <- index[, vect[key1] * vect[key2]]
  
  # Both hadamard and the global key pairs are in the same order, so we just copy over y
  hadamard$y <- key.pairs$y
  
  return(hadamard)
}

cos.relational.matrix <- function(edges, matching, key.pairs, model) {
  
  # Get the data based on matching type and edge directioness
  vect <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.vectors.tsv'), header = F, sep = "\t")
  dict <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.dict.tsv'), header = T, sep = "\t")
  dict$index <- 1:nrow(dict)
  dict$type <- NULL
  
  index <- 
    unique(c(
      merge(key.pairs, dict, by.x = 'key1', by.y = 'key', sort = F)$index,
      merge(key.pairs, dict, by.x = 'key2', by.y = 'key', sort = F)$index
    ))
  
  vect <- t(vect[index])
  cos <- cpp_RelationalMatrix(vect, coefficients(glm.cos.gup$finalModel))
  
  return(cos)
}

summaryFbeta <- function(data, lev = NULL, model = NULL) {
  cpp_Performance(pred = data$pred, obs = data$obs, posValue = 1)
}

performance <- function(model, newData) {
  cpp_Performance(pred = predict(model, newData[, !'y']), obs = newData$y, posValue = 1)
}

performance.glm <- function(model, newData) {
  cpp_Performance(pred = predict(model, newData[, !'y'], type ='response') > 0.5, obs = newData$y, posValue = 1)
}

# Tunes an SVM according some preset settings
tune.svm <- function(pairs) {
  
  control <- trainControl(method = 'cv', # cross validation
                          number = 10,   # nr of cv sets
                          classProbs = T, # return prediction probabilities along with predicted classes
                          summaryFunction = summaryFbeta,
                          returnData = F, # disable return of training data e.g. for big data sets
                          allowParallel = T)
  
  model <- train(form = y ~ ., 
                 data = pairs, 
                 method = 'svmRadial', 
                 metric = 'F0.5',
                 tuneGrid = data.frame(sigma = unname(sigest(y~., data = pairs)[2]), C = 2^(2:4)),
                 #tuneLength = 8, # Causes C = 2^((1:8) - 5)
                 trControl = control)
  
  return(model$finalModel)
}

tune.glm <- function(trainData) {
  stats::glm(y~sim + I(sim^2), trainData, family='binomial')
}

tune.lasso.glm <- function(trainData) {
  
  control <- trainControl(method = 'cv', # cross validation
                          number = 10,   # nr of cv sets
                          classProbs = T, # return prediction probabilities along with predicted classes
                          summaryFunction = summaryFbeta,
                          returnData = F, # disable return of training data e.g. for big data sets
                          allowParallel = T)
  model <- train(form = y ~ ., 
                 data = trainData, 
                 method = 'glmnet', 
                 metric = 'F0.5',
                 family = 'binomial',
                 trControl = control)
  
  return(model)
}

tune.random.forest <- function(trainData) {
  
  control <- trainControl(method = 'cv', # cross validation
                          number = 10,   # nr of cv sets
                          classProbs = T, # return prediction probabilities along with predicted classes
                          summaryFunction = summaryFbeta,
                          returnData = F, # disable return of training data e.g. for big data sets
                          allowParallel = T)
  model <- train(form = y ~ ., 
                 data = trainData, 
                 method = 'rf', 
                 metric = 'F0.5',
                 trControl = control)
  
  return(model)
}

# Train a number of SVM's, each with a different train-test partition
train.multiple.cos <- function(pairs, edges, matching, n, p) {
  
  cos <- cos.data(edges, matching, pairs)
  pblapply(1:n, function(x) {
    split <- partition.pairs(cos, p)
    glm.fit <- stats::glm(y~.,split$train, family='binomial')
    #glm.fit <- suppressWarnings(tune.glm(split$train))
    return(performance.glm(glm.fit, split$test))
  })
}

box.plot <- function(extract.function) {
  
  FUN <- match.fun(extract.function)
  
  dat <- data.table(
    de = FUN(list.glm.cos.gde),
    dp = FUN(list.glm.cos.gdp),
    ue = FUN(list.glm.cos.gue),
    up = FUN(list.glm.cos.gup),
    ue = FUN(list.glm.cos.ghe),
    up = FUN(list.glm.cos.ghp)
  )
  
  colnames(dat) <- c('Directed Exact', 'Directed Partial', 'Undirected Exact', 'Undirected Partial', 'Hybrid Exact', 'Hybrid Partial')
  
  dat <- melt(dat, measure.vars = colnames(dat))
  
  ggplot(dat, aes(x=variable, y = value, fill = variable)) + 
    geom_boxplot() + 
    xlab('') +
    ylab('Cosine Similarity') +
    ylim(min(dat$value), max(dat$value)) + 
    theme_bw() + 
    theme(axis.text=element_text(size=20), axis.title=element_text(size=20), legend.position = 'none') + 
    coord_flip()
}

density.plot <- function(extract.function) {
  
  FUN <- match.fun(extract.function)
  
  dat <- data.table(
    gde = FUN(list.glm.cos.gde),
    gdp = FUN(list.glm.cos.gdp),
    gue = FUN(list.glm.cos.gue),
    gup = FUN(list.glm.cos.gup)
  )
  
  colnames(dat) <- c('Directed Exact', 'Directed Partial', 'Undirected Exact', 'Undirected Partial')
  
  ggplot(melt(dat, measure.vars = colnames(dat))) + 
    geom_density(aes(x=value, fill = variable)) + 
    xlab('') +
    ylab('') +
    xlim(0.45, 0.85) +
    theme_bw() +
    theme(axis.text=element_text(size=20), legend.position = 'none')
}


cos.sim.plot <- function(edges, matching, key.pairs) {
  
  # Get the data based on matching type and edge directioness
  vect <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.vectors.tsv'), header = F, sep = "\t")
  dict <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.dict.tsv'), header = T, sep = "\t")
  dict$index <- 1:nrow(dict)
  dict$type <- NULL
  
  # Vect and dict are in the same order, but key pairs is not!
  # Using the keys in dict, merge with the keys in the global key pairs
  index <- data.table(
    key1 = merge(key.pairs, dict, by.x = 'key1', by.y = 'key', sort = F)$index,
    key2 = merge(key.pairs, dict, by.x = 'key2', by.y = 'key', sort = F)$index
  )
  
  vect <- t(vect)
  
  # Now we know the index in vect for each global key pair
  cos <- data.table(sim = apply(index, 1, function(i) {
    cpp_CosineSimilarity(vect[,i['key1']], vect[,i['key2']])
  }))
  
  # Both hadamard and the global key pairs are in the same order, so we just copy over y
  cos$y <- key.pairs$y
  levels(cos$y) <- c('positive', 'negative')
  
  ggplot(cos) + 
    geom_density(aes(x = sim, fill = y), alpha = 0.5) + 
    xlab('') + 
    ylab('') +
    xlim(-1,1) + 
    labs(fill = 'pairs') +
    theme_bw() +
    scale_fill_manual(values = c('blue', 'red')) +
    theme(axis.text=element_text(size=20), legend.position = 'none')
}

cos.data <- function(edges, matching, key.pairs) {
  
  # Get the data based on matching type and edge directioness
  vect <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.vectors.tsv'), header = F, sep = "\t")
  dict <- fread(file = paste0('data/saa/saa.', 'glove', '.', edges , '.', matching,'.dict.tsv'), header = T, sep = "\t")
  dict$index <- 1:nrow(dict)
  dict$type <- NULL
  
  # Vect and dict are in the same order, but key pairs is not!
  # Using the keys in dict, merge with the keys in the global key pairs
  index <- data.table(
    key1 = merge(key.pairs, dict, by.x = 'key1', by.y = 'key', sort = F)$index,
    key2 = merge(key.pairs, dict, by.x = 'key2', by.y = 'key', sort = F)$index
  )
  
  vect <- t(vect)
  
  cos <- data.table(sim = apply(index, 1, function(i) {
    cpp_CosineSimilarity(vect[,i['key1']], vect[,i['key2']])
  }))
  
  # Both hadamard and the global key pairs are in the same order, so we just copy over y
  cos$y <- key.pairs$y
  levels(cos$y) <- c('positive', 'negative')
  
  # Modify the factor 'negative' = 0 and 'positive' = 1, for stats::glm
  cos$y <- 2 - as.integer(cos$y)
  
  return(cos)
}


###############################################################
#                    CREATE GLOBAL PAIRS                      #
###############################################################

load(file='pairs.RData')
#pairs <- get.key.pairs()


###############################################################
#                      COMPARE MODELS                         #
###############################################################

load(file='partition.RData')
#partition <- caret::createDataPartition(pairs$y, p = 0.75, list = F)

###############################################################
#                  COSINE SIMILARITY LR                       #
###############################################################

cos.sim.plot('directed', 'exact', pairs)
cos.sim.plot('directed', 'partial', pairs)
cos.sim.plot('undirected', 'exact', pairs)
cos.sim.plot('undirected', 'partial', pairs)
cos.sim.plot('hybrid', 'exact', pairs)
cos.sim.plot('hybrid', 'partial', pairs)

cos.gde <- partition.pairs(cos.data('directed', 'exact', pairs), sample = partition)
cos.gdp <- partition.pairs(cos.data('directed', 'partial', pairs), sample = partition)
cos.gue <- partition.pairs(cos.data('undirected', 'exact', pairs), sample = partition)
cos.gup <- partition.pairs(cos.data('undirected', 'partial', pairs), sample = partition)
cos.ghe <- partition.pairs(cos.data('hybrid', 'exact', pairs), sample = partition)
cos.ghp <- partition.pairs(cos.data('hybrid', 'partial', pairs), sample = partition)

glm.cos.gde <- tune.glm(cos.gde$train)
glm.cos.gdp <- tune.glm(cos.gdp$train)
glm.cos.gue <- tune.glm(cos.gue$train)
glm.cos.gup <- tune.glm(cos.gup$train)
glm.cos.ghe <- tune.glm(cos.ghe$train)
glm.cos.ghp <- tune.glm(cos.ghp$train)

round(performance.glm(glm.cos.gde, cos.gde$test), 2)
round(performance.glm(glm.cos.gdp, cos.gdp$test), 2)
round(performance.glm(glm.cos.gue, cos.gue$test), 2)
round(performance.glm(glm.cos.gup, cos.gup$test), 2)
round(performance.glm(glm.cos.ghe, cos.ghe$test), 2)
round(performance.glm(glm.cos.ghp, cos.ghp$test), 2)

###############################################################
#                     HADAMARD PAIRS                          #
###############################################################

hadamard.gde <- partition.pairs(hadamard('directed', 'exact', pairs), sample = partition)
hadamard.gdp <- partition.pairs(hadamard('directed', 'partial', pairs), sample = partition)
hadamard.gue <- partition.pairs(hadamard('undirected', 'exact', pairs), sample = partition)
hadamard.gup <- partition.pairs(hadamard('undirected', 'partial', pairs), sample = partition)
hadamard.ghe <- partition.pairs(hadamard('hybrid', 'exact', pairs), sample = partition)
hadamard.ghp <- partition.pairs(hadamard('hybrid', 'partial', pairs), sample = partition)

###############################################################
#                   HADAMARD LASSO LR                         #
###############################################################

glm.lasso.gde <- tune.lasso.glm(hadamard.gde$train)
glm.lasso.gdp <- tune.lasso.glm(hadamard.gdp$train)
glm.lasso.gue <- tune.lasso.glm(hadamard.gue$train)
glm.lasso.gup <- tune.lasso.glm(hadamard.gup$train)
glm.lasso.ghe <- tune.lasso.glm(hadamard.ghe$train)
glm.lasso.ghp <- tune.lasso.glm(hadamard.ghp$train)

round(performance(glm.lasso.gde, hadamard.gde$test), 2)
round(performance(glm.lasso.gdp, hadamard.gdp$test), 2)
round(performance(glm.lasso.gue, hadamard.gue$test), 2)
round(performance(glm.lasso.gup, hadamard.gup$test), 2)
round(performance(glm.lasso.ghe, hadamard.ghe$test), 2)
round(performance(glm.lasso.ghp, hadamard.ghp$test), 2)

###############################################################
#                 HADAMARD RANDOM FOREST                      #
###############################################################

rf.gde <- tune.random.forest(hadamard.gde$train)
rf.gdp <- tune.random.forest(hadamard.gdp$train)
rf.gue <- tune.random.forest(hadamard.gue$train)
rf.gup <- tune.random.forest(hadamard.gup$train)
rf.ghe <- tune.random.forest(hadamard.ghe$train)
rf.ghp <- tune.random.forest(hadamard.ghp$train)

round(performance(rf.gde, hadamard.gde$test), 2)
round(performance(rf.gdp, hadamard.gdp$test), 2)
round(performance(rf.gue, hadamard.gue$test), 2)
round(performance(rf.gup, hadamard.gup$test), 2)
round(performance(rf.ghe, hadamard.ghe$test), 2)
round(performance(rf.ghp, hadamard.ghp$test), 2)

###############################################################
#                      HADAMARD SVM                           #
###############################################################

svm.gde <- tune.svm(hadamard.gde$train)
svm.gdp <- tune.svm(hadamard.gdp$train)
svm.gue <- tune.svm(hadamard.gue$train)
svm.gup <- tune.svm(hadamard.gup$train)
svm.ghe <- tune.svm(hadamard.ghe$train)
svm.ghp <- tune.svm(hadamard.ghp$train)

round(performance(svm.gde, hadamard.gde$test), 2)
round(performance(svm.gdp, hadamard.gdp$test), 2)
round(performance(svm.gue, hadamard.gue$test), 2)
round(performance(svm.gup, hadamard.gup$test), 2)
round(performance(svm.ghe, hadamard.ghe$test), 2)
round(performance(svm.ghp, hadamard.ghp$test), 2)

###############################################################
#                 METRIC VARIANCE ANALYSIS                    #
###############################################################

list.glm.cos.gde <- train.multiple.cos(pairs, 'directed',   'exact',   n = 10000, p = 0.0163)
list.glm.cos.gdp <- train.multiple.cos(pairs, 'directed',   'partial', n = 10000, p = 0.0163)
list.glm.cos.gue <- train.multiple.cos(pairs, 'undirected', 'exact',   n = 10000, p = 0.0163)
list.glm.cos.gup <- train.multiple.cos(pairs, 'undirected', 'partial', n = 10000, p = 0.0163)
list.glm.cos.ghe <- train.multiple.cos(pairs, 'hybrid',    'exact',   n = 10000, p = 0.0163)
list.glm.cos.ghp <- train.multiple.cos(pairs, 'hybrid',    'partial', n = 10000, p = 0.0163)

box.plot(function(list) {
  sapply(list, function(x){x['F0.5']})
})




