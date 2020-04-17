
library(pbapply)
library(data.table)
library(stringr)
library(kernlab)
library(caret)
library(ROCR)
library(Rcpp)
library(RcppProgress)
library(ggplot2)

rm(list=lsf.str())
Rcpp::sourceCpp('src/CppFunctions.cpp')

get.pairs <- function(type, edges, matching, p = 0.5, progress = F ) {
  
  # Get the data based on matching type and edge directioness
  vect <- fread(file = paste0('data/saa/saa.', type, '.', edges , '.', matching,'.vectors.tsv'), header = F, sep = "\t")
  dict <- fread(file = paste0('data/saa/saa.', type, '.', edges , '.', matching,'.dict.tsv'), header = T, sep = "\t")
  dict$type <- NULL
  
  # Get ground truth data
  truth <- fread('data/saa/saa.ground.truth.tsv', header = T, sep = '\t', blank.lines.skip = T)
  truth <- truth[!is.na(truth$ILN_GROUP)]
  truth$key <- paste0(truth$DATASET, truth$RESOURCE_URI)
  truth$clstID <- as.integer(as.factor(truth$CLUSTER_ID))
  truth$clstID <- as.integer(paste0(truth$clstID, truth$ILN_GROUP))
  # Clear all unused columns
  truth$STATUS <- NULL
  truth$CLUSTER_COUNT <- NULL
  truth$DATASET <- NULL
  truth$CLUSTER_SIZE <- NULL
  truth$ILN_GROUP <- NULL
  truth$CLUSTER_ID <- NULL
  truth$RESOURCE_URI <- NULL

  dict <- merge(dict, truth, sort = F, all.x = T)
  dict$clstID[is.na(dict$clstID)] <- -1
  
  pairs <- transpose(setDT(createHadamardPairs(transpose(vect), dict$clstID, display_progress = progress)))
  
  # The last column is the class for a given pair (1 or 0)
  colnames(pairs)[ncol(pairs)] <- 'y'
  pairs$y <- as.factor(pairs$y)
  levels(pairs$y) <- c('no', 'yes')
  pairs$y <- relevel(pairs$y, 'yes')
  
  pairs <- unique(pairs)
  
  # Clean up
  rm(truth, vect, dict)
  
  partition <- createDataPartition(y = pairs$y, p = p, list = F)
  
  pairs.split <- list()
  
  pairs.split$train <- pairs[partition]
  pairs.split$test <- pairs[-partition]
  
  return(pairs.split)
}

tune.svm <- function(pairs) {
  control <- trainControl(method = 'cv', # cross validation
                          number = 10,   # nr of cv sets
                          classProbs = T, # return prediction probabilities along with predicted classes
                          summaryFunction = twoClassSummary,
                          returnData = F, # disable return of training data e.g. for big data sets
                          allowParallel = T)
  
  
  
  model <- train(form = y ~ ., 
                 data = pairs, 
                 method = 'svmRadial', 
                 metric = 'ROC',
                 tuneGrid = data.frame(sigma = unname(sigest(y~., data = pairs)[2]), C = 2^(0:3)),
                 #tuneLength = 8, # Causes C = 2^((1:8) - 5)
                 #preProcess = c('center', 'scale'), # Doing this makes things much worse!
                 trControl = control)
  
  return(model$finalModel)
}

cm <- function(svm, pairs) {
  confusionMatrix(predict(svm, pairs[, !'y']), pairs$y)
}

fscore <- function(svm, pairs) {
  caret::F_meas(predict(svm, pairs[, !'y']), pairs$y)
}

recall <- function(svm, pairs) {
  caret::recall(predict(svm, pairs[, !'y']), pairs$y)
}

precision <- function(svm, pairs) {
  caret::precision(predict(svm, pairs[, !'y']), pairs$y)
}

###############################################################
#                    SENSITIVITY ANALYSIS                     #
###############################################################

train.multiple <- function(type, edges, matching, n, p = 0.5) {
  pblapply(1:n, function(x) {
    pairs <- get.pairs(type, edges, matching, p = p, progress = F)
    svm <- suppressWarnings(tune.svm(pairs$train))
    list(svm = svm, cm = cm(svm, pairs$test))
  })
}

std <- function(x) sd(x)/sqrt(length(x))

extract.sensitivity <- function(list) {
  sapply(list, function(x){x$cm$byClass[1]})
}

list.gde <- train.multiple('glove', 'directed',   'exact',   n = 100, p = 0.1)
list.gdp <- train.multiple('glove', 'directed',   'partial', n = 100, p = 0.1)
list.gue <- train.multiple('glove', 'undirected', 'exact',   n = 100, p = 0.1)
list.gup <- train.multiple('glove', 'undirected', 'partial', n = 100, p = 0.1)

std(extract.sensitivity(list.gde))
std(extract.sensitivity(list.gdp))
std(extract.sensitivity(list.gue))
std(extract.sensitivity(list.gup))

sens <- data.table(
  gde = extract.sensitivity(list.gde),
  gdp = extract.sensitivity(list.gdp),
  gue = extract.sensitivity(list.gue),
  gup = extract.sensitivity(list.gup)
)

ggplot(melt(sens, measure.vars = colnames(sens))) + 
  geom_density(aes(x=value, fill = variable))

###############################################################
#                    GLOVE DIRECTED EXACT                     #
###############################################################

pairs.gde <- get.pairs('glove', 'directed', 'exact', p = 0.75, progress = T)
svm.gde <- tune.svm(pairs.gde$train)
cm.gde <- cm(svm.gde, pairs.gde$test)


###############################################################
#                    GLOVE DIRECTED PARTIAL                   #
###############################################################

pairs.gdp <- get.pairs('glove', 'directed', 'partial', p = 0.75, progress = T)
svm.gdp <- tune.svm(pairs.gdp$train)
cm.gdp <- cm(svm.gdp, pairs.gdp$test)

###############################################################
#                    GLOVE UNDIRECTED EXACT                     #
###############################################################

pairs.gue <- get.pairs('glove', 'undirected', 'exact', p = 0.75, progress = T) 
svm.gue <- tune.svm(pairs.gue$train)
cm.gue <- cm(svm.gue, pairs.gue$test)


###############################################################
#                    GLOVE UNDIRECTED PARTIAL                   #
###############################################################

pairs.gup <- get.pairs('glove', 'undirected', 'partial', p = 0.75, progress = T)
svm.gup <- tune.svm(pairs.gup$train)
cm.gup <- cm(svm.gup, pairs.gup$test)


