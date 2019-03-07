library(data.table)
library(ggplot2)
library(Rtsne)

bca.type <- 'semantic'

# Load in the data
vectors <- fread(paste0('data/cityarchives.',bca.type,'.amsgrad.50.vectors.txt'))
keys <- fread(paste0('data/cityarchives.',bca.type,'.amsgrad.50.dict.txt'))

# Only keep the records for URI's
uris <- keys$V2 == 0
vectors <- vectors[uris]
keys <- keys[uris]$V1

# Clean up
rm(uris)

# Label the keys to make sure the clusters we find are meaningful
labels <- data.table(
  ecartico_person = grepl('ecartico/persons', keys),
  ecartico_marriage = grepl('ecartico/marriage', keys),
  baptism_record = grepl('record/IndexOpDoopregister', keys),
  baptism_person = grepl('person/IndexOpDoopregister', keys),
  marriage_record =  grepl('record/IndexOpOndertrouwregister', keys),
  marriage_person =  grepl('person/IndexOpOndertrouwregister', keys),
  burial_record = grepl('record/IndexOpBegraafregistersVoor1811', keys),
  burial_person = grepl('person/IndexOpBegraafregistersVoor1811', keys),
  inventaris = grepl('inventaris', keys)
)
# Mark all other keys as 'other'
labels$other <- apply(labels, 1, function(x){!any(x)})
# Create a vector with character labels
labels <- apply(labels, 1, function(x){
  colnames(labels)[which(x)]
})
# We only have a few labels, so turn into factor
labels <- as.factor(labels)

fwrite(vectors, file = paste0('output/cityarchives.',bca.type,'.tsv'), sep = "\t", col.names = F, row.names = F, quote = F)
fwrite(data.table(keys, labels), file = paste0('output/cityarchives.',bca.type,'.labels.tsv'), sep = "\t", row.names = F)


# Use principal component analysis to reduce the number of dimensions
pca <- prcomp(vectors)

# We are fine with using the principal components that explain min.var of the variance
min.var <- 0.9
# Calculate variance
pca.var <- pca$sdev^2
# Take the cumulutive sum of proportional variance
pca.cum.var <- cumsum(pca.var/sum(pca.var))

# For fun, let's view a scree-like plot of this pca object
plot(pca.var/sum(pca.var), 
     xlab = 'Principal Component',
     ylab = 'Proportion of Variance Explained',
     type = 'b')

# Find the first component that meets the min.var mark
pca.min <- min(which(pca.cum.var >= min.var))

# Transform our original vectors to pca space, taking only the components necessary to reach min.var
vectors.pc <- predict(pca, vectors)[,1:pca.min]

# Clean up some more
rm(pca.cum.var, pca.var, min.var)

# To visualize our data, perform tsne. 
# We set pca to false because we already did that ourselves
# and num_threads to 0, which means use all available cores
vis <- Rtsne(X = vectors.pc, dims = 2, pca = F, num_threads = 0)

# Plot the result
ggplot(data.table(vis$Y)) +
  geom_point(aes(x = V1, y = V2, col = labels))

