# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# choose a work directory
mywd = "C:/ ... /Downloads"
# mywd = "C:/ ... /Downloads"
setwd(mywd)

# create a name for a .txt file to log progress information while parallel processing
myfile = "log.txt"
file.create(myfile)

# cross validation folds
K = 2

# cross validation replications per fold
R = 5

# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# these are the packages i use

# data handling
require(data.table)
require(stringr)
require(tm)
require(stringdist)
require(gtools)
require(psych)

# plotting
require(VIM)
require(ggplot2)
require(gridExtra)
require(scales)
require(corrplot)
require(factoextra)

# modeling
require(forecast)
require(ranger)
require(e1071)
require(glmnet)
require(pROC)
require(caret)
require(cvTools)
require(SuperLearner)
require(xgboost)
require(h2o)
require(MLmetrics)

# parallel computing
require(foreach)
require(parallel)
require(doSNOW)
require(rlecuyer)

}

# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  dat = data.frame(dat)
  
  column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
  data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
  levels = sapply(1:ncol(dat), function(i) length(levels(dat[,i])))
  
  return(data.frame(column, data.type, levels))
}

# ---- a qualitative color scheme ---------------------------------------------------

mycolors = function(n)
{
  require(grDevices)
  return(colorRampPalette(c("#e41a1c", "#0099ff", "#4daf4a", "#984ea3", "#ff7f00", "#ff96ca", "#a65628"))(n))
}

# ---- generates a logarithmically spaced sequence ----------------------------------

lseq = function(from, to, length.out)
{
  return(exp(seq(log(from), log(to), length.out = length.out)))
}

# ---- builds a square confusion matrix ---------------------------------------------

confusion = function(ytrue, ypred)
{
  require(gtools)
  
  # make predicted and actual vectors into factors, if they aren't already
  if(class(ytrue) != "factor") ytrue = factor(ytrue)
  if(class(ypred) != "factor") ypred = factor(ypred)
  
  # combine their levels into one unique set of levels
  common.levels = mixedsort(unique(c(levels(ytrue), levels(ypred))))
  
  # give each vector the same levels
  ytrue = factor(ytrue, levels = common.levels)
  ypred = factor(ypred, levels = common.levels)
  
  # return a square confusion matrix
  return(table("Actual" = ytrue, "Predicted" = ypred))
}

# ---- runs goodness of fit tests across all columns of two data sets ---------------

sample.test = function(dat.sample, dat.remain, alpha = 0.5)
{
  # set up the types() function
  # this function extracts the column names, data types, and number of factor levels for each column of a data set
  types = function(dat)
  {
    dat = data.frame(dat)
    
    column = sapply(1:ncol(dat), function(i) colnames(dat)[i])
    data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
    levels = sapply(1:ncol(dat), function(i) length(levels(dat[,i])))
    
    return(data.frame(column, data.type, levels))
  }
  
  # make the data sets into data frames
  dat.sample = data.frame(dat.sample)
  dat.remain = data.frame(dat.remain)
  
  # get the data types of the data sets
  sample.types = types(dat.sample)
  remain.types = types(dat.remain)
  
  # ensure these data sets are identical
  if(identical(sample.types, remain.types))
  {
    # extract the column postion of factor variables
    factor.id = which(sample.types$data.type == "factor")
    
    # extract the column postion of numeric variables
    numeric.id = which(sample.types$data.type == "numeric" | sample.types$data.type == "integer")
    
    # get the p-values for the factor variables
    factor.test = lapply(factor.id, function(i)
    {
      # get the probability of each level of a factor occuring in dat.remain
      prob = as.numeric(table(dat.remain[,i]) / length(dat.remain[,i]))
      
      # get the frequency of each level of a factor occuring in dat.sample
      tab = table(dat.sample[,i])
      
      # perform a chi.sq test to reject or fail to reject the null hypothesis
      # the null: the observed frequency (tab) is equal to the expected count (prob)
      p.val = chisq.test(tab, p = prob)$p.value
      
      # determine if these variables are expected to come from the same distribution
      same.distribution = p.val > alpha
      
      # build a summary for variable i
      output = data.frame(variable = colnames(dat.sample)[i],
                          class = "factor",
                          gof.test = "chisq.test",
                          p.value = p.val,
                          alpha = alpha,
                          same.distribution = same.distribution)
      
      return(output)
    })
    
    # merge the list of rows into one table
    factor.test = do.call("rbind", factor.test)
    
    # get the p-values for the numeric variables
    numeric.test = lapply(numeric.id, function(i)
    {
      # perform a ks test to reject or fail to reject the null hypothesis
      # the null: the two variables come from the same distribution
      p.val = ks.test(dat.sample[,i], dat.remain[,i])$p.value
      
      # determine if these variables are expected to come from the same distribution
      same.distribution = p.val > alpha
      
      # build a summary for variable i
      output = data.frame(variable = colnames(dat.sample)[i],
                          class = "numeric",
                          gof.test = "ks.test",
                          p.value = p.val,
                          alpha = alpha,
                          same.distribution = same.distribution)
      
      return(output)
    })
    
    # merge the list of rows into one table
    numeric.test = do.call("rbind", numeric.test)
    
    # combine the test results into one table
    output = rbind(factor.test, numeric.test)
    
    return(output)
    
  } else
  {
    print("dat.sample and dat.remain must have the same:\n
          1. column names\n
          2. data class for each column\n
          3. number of levels for each factor column")
  }
  }

# ---- creates an array for spliting up rows of a data set for cross validation -----

cv.folds = function(n, K, R, seed)
{
  # load required packages
  require(cvTools)
  require(data.table)
  
  # set the seed for repeatability
  set.seed(seed)
  
  # create the folds for repeated cross validation
  cv = cvFolds(n = n, K = K, R = R)
  
  # extract the fold id (which) and replication id (subsets)
  cv = data.table(cbind(cv$which, cv$subsets))
  
  # rename columns accordingly
  cv.names = c("fold", paste0("rep", seq(1:R)))
  setnames(cv, cv.names)
  
  # create the combinations of folds and replications
  # this is to make sure each fold is a test set once, per replication
  comb = expand.grid(fold = 1:K, rep = 1:R)
  
  # create a list, where each element is also a list where an element indicates which observations are in the training set and testing set for a model
  cv = lapply(1:nrow(comb), function(i)
  {
    # create the testing set
    testing = cv[fold == comb$fold[i]][[comb$rep[i] + 1]]
    
    # create the training set
    training = cv[fold != comb$fold[i]][[comb$rep[i] + 1]]
    
    # return the results in a list
    return(list(train = training, test = testing))
  })
  
  return(cv)
}

# ---- fast missing value imputation by chained random forests ----------------------

# got this from:
# https://github.com/mayer79/missRanger/blob/master/R/missRanger.R

missRanger <- function(data, maxiter = 10L, pmm.k = 0, seed = NULL, ...)
{
  cat("Missing value imputation by chained random forests")
  
  data = data.frame(data)
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  allVars <- names(which(sapply(data, function(z) (is.factor(z) || is.numeric(z)) && any(!is.na(z)))))
  
  if (length(allVars) < ncol(data)) {
    cat("\n  Variables ignored in imputation (wrong data type or all values missing: ")
    cat(setdiff(names(data), allVars), sep = ", ")
  }
  
  stopifnot(length(allVars) > 1L)
  data.na <- is.na(data[, allVars, drop = FALSE])
  count.seq <- sort(colMeans(data.na))
  visit.seq <- names(count.seq)[count.seq > 0]
  
  if (!length(visit.seq)) {
    return(data)
  }
  
  k <- 1L
  predError <- rep(1, length(visit.seq))
  names(predError) <- visit.seq
  crit <- TRUE
  completed <- setdiff(allVars, visit.seq)
  
  while (crit && k <= maxiter) {
    cat("\n  missRanger iteration ", k, ":", sep = "")
    data.last <- data
    predErrorLast <- predError
    
    for (v in visit.seq) {
      v.na <- data.na[, v]
      
      if (length(completed) == 0L) {
        data[, v] <- imputeUnivariate(data[, v])
      } else {
        fit <- ranger(formula = reformulate(completed, response = v), 
                      data = data[!v.na, union(v, completed)],
                      ...)
        pred <- predict(fit, data[v.na, allVars])$predictions
        data[v.na, v] <- if (pmm.k) pmm(fit$predictions, pred, data[!v.na, v], pmm.k) else pred
        predError[[v]] <- fit$prediction.error / (if (fit$treetype == "Regression") var(data[!v.na, v]) else 1)
        
        if (is.nan(predError[[v]])) {
          predError[[v]] <- 0
        }
      }
      
      completed <- union(completed, v)
      cat(".")
    }
    
    cat("done")
    k <- k + 1L
    crit <- mean(predError) < mean(predErrorLast)
  }
  
  cat("\n")
  if (k == 2L || (k == maxiter && crit)) data else data.last
}

}

# -----------------------------------------------------------------------------------
# ---- Prepare Data -----------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Checking Data Types ----------------------------------------------------------

{

# import the data
# for column descriptions see: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
train = data.table(read.csv("train.csv", na.strings = ""))
test = data.table(read.csv("test.csv", na.strings = ""))

# lets check out train
train
types(train)

# update columns that should be treated as a different data type
train[, LotFrontage := as.numeric(as.character(LotFrontage))]
train[, MasVnrArea := as.numeric(as.character(MasVnrArea))]
train[, GarageYrBlt := as.numeric(as.character(GarageYrBlt))]
train[, MoSold := factor(MoSold, levels = sort(unique(MoSold)))]

# lets check out test
# extract the data types from train
check.train = data.table(types(train))
setnames(check.train, c("column", "train.data.type", "train.levels"))

# extract the data types from test
check.test = data.table(types(test))
setnames(check.test, c("column", "test.data.type", "test.levels"))

# join the data types of train onto test 
# this is to make sure that test has the same data types as train for each column
setkey(check.train, column)
setkey(check.test, column)
check.types = data.table(check.test[check.train])
check.types

# update columns in test that should be treated as a different data type
test[, BsmtFinSF1 := as.numeric(as.character(BsmtFinSF1))]
test[, BsmtFinSF2 := as.numeric(as.character(BsmtFinSF2))]
test[, BsmtFullBath := as.numeric(as.character(BsmtFullBath))]
test[, BsmtHalfBath := as.numeric(as.character(BsmtHalfBath))]
test[, BsmtUnfSF := as.numeric(as.character(BsmtUnfSF))]
test[, GarageArea := as.numeric(as.character(GarageArea))]
test[, GarageCars := as.numeric(as.character(GarageCars))]
test[, GarageYrBlt := as.numeric(as.character(GarageYrBlt))]
test[, LotFrontage := as.numeric(as.character(LotFrontage))]
test[, MasVnrArea := as.numeric(as.character(MasVnrArea))]
test[, MoSold := factor(MoSold, levels = sort(unique(MoSold)))]
test[, TotalBsmtSF := as.numeric(as.character(TotalBsmtSF))]

# combine train and test to make sure all factors share the same levels
dat = data.table(rbind(train[,!"SalePrice"], test))

# split up train and test
ID = max(train$Id)
train = data.table(cbind(dat[1:ID], SalePrice = train$SalePrice))
test = data.table(dat[(ID + 1):nrow(dat)])

# remove objects we no longer need
rm(check.train, check.test, check.types, ID)

# free memory
gc()

}

# ---- Check for Missing Values -----------------------------------------------------

{

# lets check out if there are any missing values (NA's) in the train
train.missing = aggr(train, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)

# extract the names of variables with missing values
train.missing = data.table(train.missing$missings)
train.missing = train.missing[Count > 0, Variable]

# lets check out if there are any missing values (NA's) in test
test.missing = aggr(test, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)

# extract the names of variables with missing values
test.missing = data.table(test.missing$missings)
test.missing = test.missing[Count > 0, Variable]

# missing variables
missing.vars = unique(c(train.missing, test.missing))
missing.vars

# lets check out what data type these missing variables are
missing.types = data.table(types(train))
missing.types[column %in% missing.vars]

# remove objects we no longer need
rm(train.missing, test.missing, missing.types)

# free memory
gc()

}

# ---- Imputations ------------------------------------------------------------------

{

# create a copy of dat
dat.impute = data.table(dat)

# remove any factor variables with more than 10 levels becuase thats too many (ie. too many degrees of freedom)
# remove Id becuase it is just an ID column, not a predictor
large.factors = data.table(types(dat.impute))
large.factors = as.character(large.factors[levels > 10, column])
dat.impute[, c("Id", large.factors) := NULL]

# choose the number of workers for parallel processing
workers = 15

# setup seeds for parallel processing
set.seed(42)
seeds = sample(1:1000, 6)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
clusterSetupRNGstream(cl, seed = seeds)
registerDoSNOW(cl)

# use a random forest to impute the missing values
rf.impute = missForest(dat.impute, ntree = 500, nodesize = c(5, 1), parallelize = "forests")

# end parallel processing
stopCluster(cl)

# free memory
gc()

# extract the imputed data set
dat.impute = rf.impute$ximp

# only keep variables in missing.vars for dat.impute
dat.impute = dat.impute[, missing.vars, with = FALSE]

# give dat.impute its PassengerId column
dat.impute[, Id := 1:nrow(dat.impute)]

# remove variables in missing.vars from train and test
train = train[, !missing.vars, with = FALSE]
test = test[, !missing.vars, with = FALSE]

# make Id the key column in dat.impute, train, and test for joining purposes
setkey(dat.impute, Id)
setkey(train, Id)
setkey(test, Id)

# join dat.impute onto train and onto test
train = dat.impute[train]
test = dat.impute[test]

# lets remove objects we no longer need
rm(dat.impute, dat, missing.vars, rf.impute, seeds, workers, cl, large.factors)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Feature Engineering ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Scaling ----------------------------------------------------------------------

{

# combine train and test so we can scale the data
dat = data.table(rbind(train[,!"SalePrice"], test))

# remove the Id column so we can scale the data
dat[, Id := NULL]

# reformat all columns to be numeric by creating dummy variables for factor columns
dat = data.table(model.matrix(~., dat)[,-1])

# scale dat so that all variables can be compared fairly
dat = data.table(scale(dat))

# give the Id column back to dat
dat[, Id := 1:nrow(dat)]

# split up dat into train and test
train = cbind(SalePrice = train$SalePrice, dat[Id %in% train$Id])
test = dat[Id %in% test$Id]

# remove the Id column from train and test
train[, Id := NULL]
test[, Id := NULL]

# remove objects we no longer need
rm(dat)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Feature Selection ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- ANOVA ------------------------------------------------------------------------

{

# build a copy of train for aov
aov.dat = data.table(train[,!"SalePrice"])

# create column names for aov.dat
# remove all `
aov.names = gsub("`", "", names(aov.dat))

# remove all spacing
aov.names = gsub(" ", "", aov.names)

# replace ":" with "."
aov.names = gsub(":", ".", aov.names)

# set the names of aov.dat
setnames(aov.dat, aov.names)

# attach SalePrice to aov.dat
aov.dat = cbind(SalePrice = train$SalePrice, aov.dat)

# ---- Cut 1: Keep variables with p-value < 0.50 -----------------------------------

# build an anova table
my.aov = aov(SalePrice ~., data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.5
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.5, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# setup aov.dat to have the variables in keep.var
aov.dat = data.table(aov.dat[, c("SalePrice", keep.var), with = FALSE])

# ---- Cut 2: Keep variables with p-value < 0.25 -----------------------------------

# build an anova table
my.aov = aov(SalePrice ~., data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.5
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.25, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# setup aov.dat to have the variables in keep.var
aov.dat = data.table(aov.dat[, c("SalePrice", keep.var), with = FALSE])

# ---- Cut 3: Keep variables with p-value < 0.10 -----------------------------------

# build an anova table
my.aov = aov(SalePrice ~., data = aov.dat)

# convert anova to a data table
my.aov = as.data.frame(summary(my.aov)[[1]])
my.aov$var = rownames(my.aov)
my.aov = data.table(my.aov)

# keep all terms that have p-value < 0.1
my.aov = my.aov[, c("var", "Pr(>F)"), with = FALSE]
setnames(my.aov, c("var", "p"))
my.aov = na.omit(my.aov)
keep.var = my.aov[p < 0.1, var]

# remove all ` in keep.var
keep.var = gsub("`", "", keep.var)

# remove all spacing in keep.var
keep.var = gsub(" ", "", keep.var)

# make sure only aov.names are in keep.var
keep.var = keep.var[which((keep.var %in% aov.names) == TRUE)]

# remove objects we no longer need
rm(aov.dat, my.aov, aov.names)

# free memory
gc()

}

# ---- Correlation ------------------------------------------------------------------

{

# build a copy of train for modeling
mod.dat = data.table(train)

# extract all potential variables
cor.dat = data.table(mod.dat[,!"SalePrice"])

# create column names for cor.dat
# remove all `
cor.names = gsub("`", "", names(cor.dat))

# remove all spacing
cor.names = gsub(" ", "", cor.names)

# replace ":" with "."
cor.names = gsub(":", ".", cor.names)

# set the names of aov.dat
setnames(cor.dat, cor.names)

# attach cor.dat to mod.dat
mod.dat = cbind(SalePrice = as.numeric(mod.dat$SalePrice), cor.dat)

# setup mod.dat and cor.dat to have the variables in keep.var
mod.dat = data.table(mod.dat[, c("SalePrice", keep.var), with = FALSE])
cor.dat = data.table(cor.dat[, keep.var, with = FALSE])

# compute correlations
cors = cor(cor.dat)
# replace any NA's with 1's
cors[is.na(cors)] = 1

# find out which variables are highly correlated (>= 0.9) and remove them
find.dat = findCorrelation(cors, cutoff = 0.9, names = TRUE)

# remove columns from mod.dat according to find.dat
if(length(find.dat) > 0) mod.dat = mod.dat[, !find.dat, with = FALSE]

}

# ---- Importance -------------------------------------------------------------------

{

# choose the number of workers and tasks for parallel processing
workers = 10
tasks = 10

# set up seeds for reproducability
set.seed(42)
seeds = sample(1:1000, tasks)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("random forest - variable importance\n")
cat(paste(workers, "workers started at", Sys.time()), "\n")
sink()

# build random forest models in parallel
var.imp = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(randomForest)
  require(data.table)
  require(caret)
  
  # build randomForest
  set.seed(seeds[i])
  mod = randomForest(SalePrice ~ ., data = mod.dat, 
                     ntree = 1000, importance = TRUE)
  
  # compute variable importance on a scale of 0 to 100
  imp = varImp(mod, scale = TRUE)
  
  # transform imp into long format
  imp = data.table(variable = rownames(imp), 
                   value = (rowSums(imp) / ncol(imp)) / 100)
  
  # add the task number to imp
  imp[, task := i]
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time()), "\n")
  sink()
  
  return(imp)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time()), "\n")
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of data tables into one table
var.imp = rbindlist(var.imp)

# average importance of variables
var.imp = var.imp[, .(value = mean(value)), by = .(variable)]

# order by importance
var.imp = var.imp[order(value, decreasing = TRUE)]

# make variable a factor for plotting purposes
var.imp[, variable := factor(variable, levels = unique(variable))]

# plot a barplot of variable importance
ggplot(var.imp, aes(x = variable, y = value, fill = value, color = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Variable", y = "Importance") +
  scale_y_continuous(labels = percent) +
  scale_fill_gradient(low = "yellow", high = "red") +
  scale_color_gradient(low = "yellow", high = "red") +
  theme_dark(15) +
  theme(legend.position = "none", axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank())

# lets only keep variables with at least 5% importance
keep.dat = gsub("`", "", var.imp[value >= 0.05, variable])
mod.dat = mod.dat[, c("SalePrice", keep.dat), with = FALSE]

# heres our variables for SalePrice
SalePrice.variables = keep.dat

}

# ---- Finalize Data ----------------------------------------------------------------

{

# setup train to have the variables in SalePrice.variables
train = data.table(mod.dat)

# setup test to have the variables in SalePrice.variables
# create column names for test
# remove all `
test.names = gsub("`", "", names(test))

# remove all spacing
test.names = gsub(" ", "", test.names)

# replace ":" with "."
test.names = gsub(":", ".", test.names)

# set the names of test
setnames(test, test.names)

# only keep the model variables
test = test[, SalePrice.variables, with = FALSE]

# remove objects we no longer need
rm(test.names, cor.names, keep.var, SalePrice.variables, cor.dat, cors, find.dat, mod.dat, keep.dat, var.imp, cl, seeds, tasks, workers)

# free memory
gc()

}

}

# -----------------------------------------------------------------------------------
# ---- Linear Regression Model ------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# extract predictors (X) and response (Y)
X = data.table(train[,!"SalePrice"])
Y = train$SalePrice

# build the cross validation folds
cv = cv.folds(n = nrow(X), K = K, R = R, seed = 42)

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our models
lm.pred = function(Xtrain, Ytrain, Xtest, Ytest)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = lm(y ~ ., data = dat)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, data.table(Xtest)))
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = length(cv)
tasks = length(cv)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("linear regression - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation
lm.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(forecast)
  
  # extract the training and test sets
  folds = cv[[i]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = lm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
lm.cv = rbindlist(lm.cv)

# summarize performance metrics for every model
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

lm.diag = lm.cv[,.(stat = factor(stat, levels = stat),
                   ME = as.vector(summary(na.omit(ME))), 
                   RMSE = as.vector(summary(na.omit(RMSE))),
                   MAPE = as.vector(summary(na.omit(MAPE))),
                   Rsquared = as.vector(summary(na.omit(Rsquared))))]

# ---- Results ----------------------------------------------------------------------

# add a model name column
lm.diag[, mod := rep("lm", nrow(lm.diag))]

# store model diagnostic results
mods.diag = data.table(lm.diag)

# build the model
set.seed(42)
lm.mod = glm(SalePrice ~ ., data = train)

# store the model
lm.list = list("mod" = lm.mod)
mods.list = list("lm" = lm.list)

# remove objects we no longer need
rm(lm.cv, lm.diag, lm.list, lm.mod, lm.pred, cl, tasks, workers, stat, X, Y)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Penalty Regression Model -----------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# extract predictors (X) and response (Y)
X = as.matrix(train[,!"SalePrice"])
Y = train$SalePrice

# glmnet offers:
# ridge penalty by setting the parameter alpha = 0
# lasso penalty by setting the parameter alpha = 1
# elastic net penalty by setting the parameter 0 < alpha < 1

# build a sequence of alpha values to test
doe = data.table(alpha = seq(0, 1, 0.05))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our models
pen.pred = function(Xtrain, Ytrain, Xtest, Ytest, alpha)
{
  # build the training model
  set.seed(42)
  mod = cv.glmnet(x = Xtrain, y = Ytrain, family = "gaussian", alpha = alpha)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, s = mod$lambda.min, Xtest))
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("penalty regression - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# compute diagonistics for each of the models in doe
pen.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(forecast)
  require(glmnet)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # perform cross validation
  output = pen.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, alpha = doe$alpha[i])
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  output[, mod := i]
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
pen.cv = rbindlist(pen.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

pen.diag = pen.cv[,.(stat = factor(stat, levels = stat),
                     ME = as.vector(summary(na.omit(ME))), 
                     RMSE = as.vector(summary(na.omit(RMSE))),
                     MAPE = as.vector(summary(na.omit(MAPE))),
                     Rsquared = as.vector(summary(na.omit(Rsquared)))),
                  by = alpha]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(pen.diag)
pen.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert pen.diag into long format for plotting purposes
DT = data.table(melt(pen.diag, measure.vars = c("ME", "RMSE", "MAPE", "Rsquared")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[abs(value) < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter out models
pen.diag[stat == "Mean" & mod %in% pen.diag[stat == "Median" & MAPE <= 12, mod]]

# model 1 looks good
set.seed(42)
pen.mod = cv.glmnet(x = X, y = Y, family = "gaussian", alpha = 0)

# extract coefficients of the chosen terms for the lambda that minimizes mean cross-validated error
pen.coef = coef(pen.mod, s = "lambda.min")
pen.coef = data.table(term = rownames(pen.coef), coefficient = as.numeric(pen.coef))

# store model diagnostic results
pen.diag = pen.diag[mod == 1]
pen.diag[, mod := rep("pen", nrow(pen.diag))]
pen.diag = pen.diag[,.(ME, RMSE, MAPE, Rsquared, stat, mod)]
mods.diag = rbind(mods.diag, pen.diag)

# store the model
pen.list = list("mod" = pen.mod, "coef" = pen.coef)
mods.list$pen = pen.list

# remove objects we no longer need
rm(pen.cv, pen.diag, pen.list, pen.mod, pen.pred, doe, DT, 
   pen.coef, X, Y, diag.plot, workers, tasks, cl, num.stats, stat, num.rows)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Gradient Boosting Model ------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# we have 7 hyperparameters of interest:
  # nrounds ~ the max number of boosting iterations
  # eta ~ the learning rate
  # max_depth ~ maximum depth of a tree
  # min_child_weight ~ minimum sum of instance weight needed in a child
  # gamma ~ minimum loss reduction required to make a further partition on a leaf node of the tree
  # subsample ~ the proportion of data (rows) to randomly sample each round
  # colsample_bytree ~ the proportion of variables (columns) to randomly sample each round

# check out this link for help on tuning:
  # https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur
  # google stuff and you'll find other approaches

# extract predictors (X) and response (Y)
X = as.matrix(train[,!"SalePrice"])
Y = train$SalePrice

# create parameter combinations to test
doe = data.table(expand.grid(nrounds = 100,
                             eta = 0.1,
                             max_depth = c(4, 6, 8, 10, 12, 14), 
                             min_child_weight = c(1, 3, 5, 7, 9, 11),
                             gamma = 0,
                             subsample = 1,
                             colsample_bytree = 1))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
gbm.pred = function(Xtrain, Ytrain, Xtest, Ytest, objective, eval_metric, eta, max_depth, nrounds, min_child_weight, gamma, subsample, colsample_bytree)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Ytrain, data = Xtrain,
                objective = objective, eval_metric = eval_metric,
                eta = eta, max_depth = max_depth,
                nrounds = nrounds, min_child_weight = min_child_weight,
                gamma = gamma, verbose = 0,
                subsample = subsample, colsample_bytree = colsample_bytree)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = Xtest))
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 6
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("gradient boosting - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
gbm.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(xgboost)
  require(forecast)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = gbm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    objective = "reg:linear", eval_metric = "rmse",
                    eta = doe$eta[i], max_depth = doe$max_depth[i], nrounds = doe$nrounds[i], 
                    min_child_weight = doe$min_child_weight[i], gamma = doe$gamma[i], 
                    subsample = doe$subsample[i], colsample_bytree = doe$colsample_bytree[i])
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
gbm.cv = rbindlist(gbm.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

gbm.diag = gbm.cv[,.(stat = factor(stat, levels = stat),
                     ME = as.vector(summary(na.omit(ME))), 
                     RMSE = as.vector(summary(na.omit(RMSE))),
                     MAPE = as.vector(summary(na.omit(MAPE))),
                     Rsquared = as.vector(summary(na.omit(Rsquared)))),
                  by = .(eta, max_depth, nrounds, min_child_weight, gamma, subsample, colsample_bytree)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(gbm.diag)
gbm.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert gbm.diag into long format for plotting purposes
DT = data.table(melt(gbm.diag, measure.vars = c("ME", "RMSE", "MAPE", "Rsquared")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[abs(value) < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
gbm.diag[stat == "Mean" & mod %in% gbm.diag[stat == "Median" & RMSE <= 32500 & mod %in% gbm.diag[stat == "Max" & RMSE <= 38500, mod], mod]]

# model 7 looks good
gbm.diag = gbm.diag[mod == 7]

# rename model to gbm
gbm.diag[, mod := rep("gbm", nrow(gbm.diag))]

# build our model
set.seed(42)
gbm.mod = xgboost(label = Y, data = X,
                  objective = "reg:linear", eval_metric = "rmse",
                  eta = 0.1, max_depth = 4, nrounds = 100, 
                  min_child_weight = 3, gamma = 0, verbose = 0,
                  subsample = 1, colsample_bytree = 1)

# store model diagnostic results
gbm.diag = gbm.diag[,.(ME, RMSE, MAPE, Rsquared, stat, mod)]
mods.diag = rbind(mods.diag, gbm.diag)

# store the model
gbm.list = list("mod" = gbm.mod)
mods.list$gbm = gbm.list

# remove objects we no longer need
rm(gbm.cv, gbm.diag, gbm.list, gbm.mod, gbm.pred, doe, DT, 
   X, Y, diag.plot, cl, workers, tasks, num.rows, num.stats, stat)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Random Forest Model ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# we have 3 hyperparameters of interest:
# ntree ~ number of decision trees to create
# nodesize ~ minimum size of terminal nodes (ie. the minimum number of data points that can be grouped together in any node of a tree)

# check out this link for help on tuning:
# https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur

# extract predictors (X) and response (Y)
X = data.table(train[,!"SalePrice"])
Y = train$SalePrice

# create parameter combinations to test
doe = data.table(expand.grid(ntree = c(500, 800, 1200),
                             nodesize = c(5, 7, 9, 14)))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
rf.pred = function(Xtrain, Ytrain, Xtest, Ytest, ntree, nodesize)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = randomForest(y ~ .,
                     data = dat,
                     ntree = ntree,
                     nodesize = nodesize)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(Xtest)))
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("random forest - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
rf.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(randomForest)
  require(forecast)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = rf.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                   ntree = doe$ntree[i], nodesize = doe$nodesize[i])
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
rf.cv = rbindlist(rf.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

rf.diag = rf.cv[,.(stat = factor(stat, levels = stat),
                   ME = as.vector(summary(na.omit(ME))), 
                   RMSE = as.vector(summary(na.omit(RMSE))),
                   MAPE = as.vector(summary(na.omit(MAPE))),
                   Rsquared = as.vector(summary(na.omit(Rsquared)))),
                by = .(ntree, nodesize)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(rf.diag)
rf.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert rf.diag into long format for plotting purposes
DT = data.table(melt(rf.diag, measure.vars = c("ME", "RMSE", "MAPE", "Rsquared")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[abs(value) < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
rf.diag[stat == "Mean" & ME <= 450 & mod %in% rf.diag[stat == "Median" & MAPE <= 11 & RMSE <= 33500, mod]]

# model 9 looks good
rf.diag = rf.diag[mod == 9]

# rename model to rf
rf.diag[, mod := rep("rf", nrow(rf.diag))]

# build the model
set.seed(42)
rf.mod = randomForest(SalePrice ~ ., data = train,
                      ntree = 1200, nodesize = 9)

# store model diagnostic results
rf.diag = rf.diag[,.(ME, RMSE, MAPE, Rsquared, stat, mod)]
mods.diag = rbind(mods.diag, rf.diag)

# store the model
rf.list = list("mod" = rf.mod)
mods.list$rf = rf.list

# remove objects we no longer need
rm(rf.cv, rf.diag, rf.list, rf.mod, rf.pred, doe, DT, X, Y, diag.plot,
   workers, tasks, cl, num.stats, num.rows, stat)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Deep Nueral Network Model ----------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# extract predictors (X) and response (Y)
X = data.table(train[,!"SalePrice"])
Y = train$SalePrice

# check out the following link to understand h2o deep learning
# http://h2o-release.s3.amazonaws.com/h2o/rel-tukey/6/docs-website/h2o-docs/booklets/R_Vignette.pdf

# we have 3 hyperparameters of interest:
# hidden ~ a vector of integers indicating the number of nodes in each hidden layer
# l1 ~ L1 norm regularization to penalize large weights (may cause many weights to become 0)
# l2 ~ L2 norm regularization to penalize large weights (may cause many weights to become small)

# set up L1 & L2 penalties
l1 = 1e-5
l2 = 1e-5

# use the same seed that we've been using for model building
seed = 42

# how many times the training data should be passed through the network to adjust path weights
epochs = 50

# the classes are imbalanced so lets set up the balance_classes and class_sampling_factors parameters
balance_classes = TRUE
class_sampling_factors = table(Y)
class_sampling_factors = as.vector(max(class_sampling_factors) / class_sampling_factors)

# choose the total number of hidden nodes
nodes = 150

# choose the hidden layer options to distribtuion nodes across
layers = 1:5

# choose whether to try varying structures for each layer (0 = No, 1 = Yes)
vary = 0

# initilize the size of doe
N = max(layers)
doe = matrix(ncol = N)

# build different ratios for distributing nodes across hidden layer options
for(n in layers)
{
  # single layer
  if(n == 1)
  {
    # just one layer
    op = c(1, rep(0, N - n))
    
    # store layer option
    doe = rbind(doe, op)
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op)
    
    # double layer
  } else if(n == 2)
  {
    # layers increase in size
    op1 = c(1:n, rep(0, N - n))
    # layers decrease in size
    op2 = c(n:1, rep(0, N - n))
    # layers are equal in size
    op3 = c(rep(1, length.out = n), rep(0, N - n))
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3)
    
    # largest multi-layer
  } else if(n == N)
  {
    # layers increase in size
    op1 = 1:n
    # layers decrease in size
    op2 = n:1
    # layers are equal in size
    op3 = rep(1, length.out = n)
    # layers oscilate in size, starting low
    op4 = rep(1:2, length.out = n)
    # layers oscilate in size, starting high
    op5 = rep(2:1, length.out = n)
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    op4 = op4 / sum(op4)
    op5 = op5 / sum(op5)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3, op4, op5)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3, op4, op5)
    
    # not the largest multi-layer
  } else
  {
    # op1 through op5 are the same as above
    op1 = c(1:n, rep(0, N - n))
    op2 = c(n:1, rep(0, N - n))
    op3 = c(rep(1, length.out = n), rep(0, N - n))
    op4 = c(rep(1:2, length.out = n), rep(0, N - n))
    op5 = c(rep(2:1, length.out = n), rep(0, N - n))
    
    # make layer ratios into proportions
    op1 = op1 / sum(op1)
    op2 = op2 / sum(op2)
    op3 = op3 / sum(op3)
    op4 = op4 / sum(op4)
    op5 = op5 / sum(op5)
    
    # store layer options
    if(vary == 1)
    {
      doe = rbind(doe, op1, op2, op3, op4, op5)
      
    } else
    {
      doe = rbind(doe, op3)
    }
    
    rownames(doe) = 0:(nrow(doe) - 1)
    rm(op1, op2, op3, op4, op5)
  }
}

rm(n, N)

# remove the first row of doe becuase it was just a dummy row to append to
doe = doe[-1,]
doe = data.frame(doe)

# add cross validation ids for each scenario in doe
doe = data.frame(rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe))))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
dnn.pred = function(Xtrain, Ytrain, Xtest, Ytest, hidden, l1, l2, epochs, seed)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # make dat and Xtest into h2o objects
  dat.h2o = as.h2o(dat)
  Xtest.h2o = as.h2o(Xtest)
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
    # the parameters that are commented out should be used only if there is a convergence issue
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = hidden,
                         l1 = l1,
                         l2 = l2,
                         epochs = epochs,
                         seed = seed,
                         # max_w2 = 10,
                         # activation = "Tanh",
                         # initial_weight_distribution = "UniformAdaptive",
                         # initial_weight_scale = 0.5,
                         variable_importances = FALSE)
  
  # make predictions with the training model using the test set
  ynew = as.data.frame(predict(mod, newdata = Xtest.h2o))$predict
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of tasks
tasks = nrow(doe)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("deep nueral network - cross validation\n")
cat(paste("task 1 started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
dnn.cv = foreach(i = 1:tasks) %do%
{
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # extract the portion of doe regarding the hidden layer structure
  doe.size = doe[,-1]
  
  # build the hidden layer structure for model i
  size = length(which(doe.size[i,] > 0))
  hidden = sapply(1:size, function(j) round(ceiling(nodes * doe.size[i,j]), 0))
  
  # build model and get prediction results
  output = dnn.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    hidden = hidden, l1 = l1, l2 = l2, epochs = epochs, seed = seed)
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i,])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# free memory
gc()

# combine the list of tables into one table
dnn.cv = rbindlist(dnn.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

dnn.diag = dnn.cv[,.(stat = factor(stat, levels = stat),
                     ME = as.vector(summary(na.omit(ME))), 
                     RMSE = as.vector(summary(na.omit(RMSE))),
                     MAPE = as.vector(summary(na.omit(MAPE))),
                     Rsquared = as.vector(summary(na.omit(Rsquared)))),
                  by = eval(paste0("X", seq(1:(ncol(doe) - 1))))]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(dnn.diag)
dnn.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert dnn.diag into long format for plotting purposes
DT = data.table(melt(dnn.diag, measure.vars = c("ME", "RMSE", "MAPE", "Rsquared")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[abs(value) < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter models
dnn.diag[stat == "Median" & mod %in% dnn.diag[stat == "Mean" & RMSE <= 33000 & mod %in% dnn.diag[stat == "Median" & MAPE <= 11, mod], mod]]

# model 7 looks good
dnn.diag = dnn.diag[mod == 7]

# rename model to dnn
dnn.diag[, mod := rep("dnn", nrow(dnn.diag))]

# build the hidden layer structure for model i
i = 7
doe.size = doe[,-1]
size = length(which(doe.size[i,] > 0))
hidden = sapply(1:size, function(j) round(ceiling(nodes * doe.size[i,j]), 0))
hidden

# recall the penalties and epochs
l1
l2
epochs

# build the model
train.h2o = as.h2o(train)
dnn.mod = h2o.deeplearning(y = "SalePrice",
                           x = colnames(X),
                           training_frame = train.h2o,
                           hidden = c(334, 334, 334),
                           l1 = 1e-05,
                           l2 = 1e-05,
                           epochs = 100,
                           seed = 42,
                           variable_importances = FALSE)

# store model diagnostic results
dnn.diag = dnn.diag[,.(ME, RMSE, MAPE, Rsquared, stat, mod)]
mods.diag = rbind(mods.diag, dnn.diag)

# store the model
dnn.list = list("mod" = dnn.mod)
mods.list$dnn = dnn.list

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# remove objects we no longer need
rm(dnn.cv, dnn.diag, dnn.list, dnn.mod, dnn.pred, i, doe, DT, output, X, Y, diag.plot, 
   layers, seed, train.h2o, epochs, hidden, l1 ,l2, nodes, size, doe.size, Xtest, Xtrain, Ytest, Ytrain,
   num.rows, num.stats, stat, tasks, folds)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Support Vector Machine Model -------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# hyperparameters of interest:
  # cost ~ controls the error penalty
    # higher values increase the error penalty and decrease the margin of seperation
    # lower values decrease the error penalty and increase the margin of seperation
  # gamma ~ controls the radius of the region of influence for support vectors
    # if too large, the region of influence of any selected support vectors would only include the support vector itself and overfit the data.
    # if too small, the region of influence of any selected support vector would include the whole training set and underfit the data
  # epsilon ~ controls the margin of tolerance where no penalty is given
    # larger values allow larger errors to be admitted in your solution. 
    # lower values means every error is penalized, so you may end with many support vectors

# check out this link for help on tuning:
  # https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur

# default value for gamma in our case:
gamma = 1 / ncol(train[,!"SalePrice"])
gamma

# extract predictors (X) and response (Y)
X = data.table(train[,!"SalePrice"])
Y = train$SalePrice

# create parameter combinations to test
doe = data.table(expand.grid(cost = lseq(0.001, 1000, 500),
                             gamma = gamma,
                             epsilon = 0.1))

# add cross validation ids for each scenario in doe
doe = rbindlist(lapply(1:length(cv), function(i) cbind(cv = rep(i, nrow(doe)), doe)))

# ---- CV ---------------------------------------------------------------------------

# build a function that will report prediction results of our model
svm.pred = function(Xtrain, Ytrain, Xtest, Ytest, cost, gamma, epsilon)
{
  # build the table for training the model
  dat = data.table(Xtrain)
  dat[, y := Ytrain]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            cost = cost,
            gamma = gamma,
            epsilon = epsilon)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(Xtest)))
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
workers = 15
tasks = nrow(doe)

# setup parallel processing
cl = makeCluster(workers, type = "SOCK", outfile = "")
registerDoSNOW(cl)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("support vector machine - cross validation\n")
cat(paste(workers, "workers started at", Sys.time(), "\n"))
sink()

# perform cross validation for each of the models in doe
svm.cv = foreach(i = 1:tasks) %dopar%
{
  # load packages we need for our tasks
  require(data.table)
  require(e1071)
  require(forecast)
  
  # extract the training and test sets
  folds = cv[[doe$cv[i]]]
  Xtrain = X[folds$train,]
  Ytrain = Y[folds$train]
  Xtest= X[folds$test,]
  Ytest = Y[folds$test]
  
  # build model and get prediction results
  output = svm.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, 
                    cost = doe$cost[i], gamma = doe$gamma[i], epsilon = doe$epsilon[i])
  
  # add columns of parameter values that define model i
  output = cbind(output, doe[i])
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
svm.cv = rbindlist(svm.cv)

# summarize performance metrics for every model in doe
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

svm.diag = svm.cv[,.(stat = factor(stat, levels = stat),
                     ME = as.vector(summary(na.omit(ME))), 
                     RMSE = as.vector(summary(na.omit(RMSE))),
                     MAPE = as.vector(summary(na.omit(MAPE))),
                     Rsquared = as.vector(summary(na.omit(Rsquared)))),
                  by = .(cost, gamma, epsilon)]

# add a column that defines model i
num.stats = length(stat)
num.rows = nrow(svm.diag)
svm.diag[, mod := sort(rep(1:(num.rows / num.stats), num.stats))]

# convert svm.diag into long format for plotting purposes
DT = data.table(melt(svm.diag, measure.vars = c("ME", "RMSE", "MAPE", "Rsquared")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod)]

# remove Inf values as these don't help
DT = data.table(DT[abs(value) < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value))) +
  geom_bar(stat = "identity", position = "dodge", color = "cornflowerblue", fill = "cornflowerblue") +
  labs(x = "Summary Statistic", y = "Value") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# ---- Results ----------------------------------------------------------------------

# lets filter out models
svm.diag[stat == "Median" & mod %in% svm.diag[stat == "Mean" & RMSE <= 33000 & ME <= 1000 & mod %in% svm.diag[stat == "Median" & MAPE <= 11 & RMSE <= 33000 & mod %in% svm.diag[stat == "Max" & MAPE <= 12, mod], mod], mod]]

# lets go with model 262
svm.diag = svm.diag[mod == 262]

# rename model to svm as this is our chosen model
svm.diag[, mod := rep("svm", nrow(svm.diag))]

# build our model
set.seed(42)
svm.mod = svm(SalePrice ~ ., data = train, cost = 1.374917, gamma = 1/39, epsilon = 0.1)

# store model diagnostic results
svm.diag = svm.diag[,.(ME, RMSE, MAPE, Rsquared, stat, mod)]
mods.diag = rbind(mods.diag, svm.diag)

# store the model
svm.list = list("mod" = svm.mod)
mods.list$svm = svm.list

# remove objects we no longer need
rm(gamma, svm.cv, svm.diag, svm.list, svm.mod, svm.pred, doe, DT, X, Y, diag.plot,
   tasks, workers, num.stats, num.rows, stat, cl)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Super Learner Model ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Set Up -----------------------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# extract predictors (X) and response (Y)
sl.X = as.matrix(train[,!"SalePrice"])
sl.Y = train$SalePrice

# create Super Learner wrappers for our models of interest

# linear regression wrapper
my.lm = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # build the training model
  set.seed(42)
  mod = lm(y ~ ., data = dat)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, data.table(newX)))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# penalty regression wrapper
my.pen = function(Y, X, newX, ...)
{
  # build the training model
  set.seed(42)
  mod = cv.glmnet(x = X, y = Y, family = "gaussian", alpha = 0)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, s = mod$lambda.min, newX))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# gradient boosting wrapper
my.gbm = function(Y, X, newX, ...)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X,
                objective = "reg:linear", eval_metric = "rmse",
                eta = 0.1, max_depth = 4, nrounds = 100, 
                min_child_weight = 3, gamma = 0, verbose = 0,
                subsample = 1, colsample_bytree = 1)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = newX))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# random forest wrapper
my.rf = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # build the training model
  set.seed(42)
  mod = randomForest(y ~ .,
                     data = dat,
                     ntree = 1200, 
                     nodesize = 9)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(newX)))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# deep neural network wrapper
my.dnn = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # make dat and newX into h2o objects
  dat.h2o = as.h2o(dat)
  newX.h2o = as.h2o(data.table(newX))
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = c(334, 334, 334),
                         l1 = 1e-05,
                         l2 = 1e-05,
                         epochs = 100,
                         seed = 42,
                         variable_importances = FALSE)
  
  # make predictions with the training model using the test set
  ynew = as.data.frame(predict(mod, newdata = newX.h2o))$predict
  
  # run garbage collection to free up space for h2o
  gc()
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# support vector machine wrapper
my.svm = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  new.dat = data.table(newX)
  
  # give dat the response variable
  dat[, y := Y]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            cost = 1.374917, 
            gamma = 1/39, 
            epsilon = 0.1)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = new.dat))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# create a library of the above wrappers
my.library = list("my.lm", "my.pen", "my.gbm", "my.rf", "my.dnn", "my.svm")

# ---- Choosing Models --------------------------------------------------------------

# build the super learner model
set.seed(42)
sl.mod = SuperLearner(Y = sl.Y, X = sl.X, family = gaussian(), SL.library = my.library, verbose = TRUE)
sl.mod

# remove lm and pen for having larger risk
my.library = my.library[-which(my.library %in% c("my.lm", "my.pen"))]

# if dnn was removed then shutdown the h2o instance
if(!("my.dnn" %in% my.library))
{
  h2o.shutdown(prompt = FALSE)
}

# build the super learner model
set.seed(42)
sl.mod = SuperLearner(Y = sl.Y, X = sl.X, family = gaussian(), SL.library = my.library, verbose = TRUE)
sl.mod

# ---- CV ---------------------------------------------------------------------------

# the snow cluster won't work for SuperLearner
# even the example page won't run properly
# so we will have to do this cross validation sequentially
# thats why all of the parallel processing related commands are commented out
# also if the dnn is in the super learner than that is already using up a significant portion of the CPU
# so this cross validation shouldn't even be considered for parallel processing unless the dnn is not in my.library

# build a function that will report prediction results of our models
sl.pred = function(Xtrain, Ytrain, Xtest, Ytest, my.library)
{
  # build the training model
  set.seed(42)
  mod = SuperLearner(Y = Ytrain, X = Xtrain, newX = Xtest, 
                     family = gaussian(), SL.library = my.library)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(mod$SL.predict)
  
  # build a table to summarize the performance of our training model
  output = data.table(cbind(t(accuracy(f = ynew, x = Ytest)[, c("ME", "RMSE", "MAPE")]),
                            cor(ynew, Ytest)^2))
  
  setnames(output, c("ME", "RMSE", "MAPE", "Rsquared"))
  
  # free memory
  gc()
  
  return(output)
}

# choose the number of workers and tasks for parallel processing
# workers = 5
tasks = length(cv)

# setup seeds for parallel processing
# set.seed(42)
# seeds = sample(1:1000, 6)

# setup parallel processing
# cl = makeCluster(workers, type = "SOCK", outfile = "")
# clusterSetupRNGstream(cl, seed = seeds)
# registerDoSNOW(cl)

# assign the prediction functions in my.library to the SuperLearner namespace
# environment(my.lm) = asNamespace("SuperLearner")
# environment(my.pen) = asNamespace("SuperLearner")
# environment(my.gbm) = asNamespace("SuperLearner")
# environment(my.rf) = asNamespace("SuperLearner")
# environment(my.dnn) = asNamespace("SuperLearner")
# environment(my.svm) = asNamespace("SuperLearner")

# copy the prediction functions in my.library to all clusters
# clusterExport(cl, varlist = my.library)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("super learner - cross validation\n")
cat(paste("task 1 started at", Sys.time(), "\n"))
sink()

# perform cross validation
sl.cv = foreach(i = 1:tasks) %do%
{
  # extract the training and test sets
  folds = cv[[i]]
  Xtrain = sl.X[folds$train,]
  Ytrain = sl.Y[folds$train]
  Xtest= sl.X[folds$test,]
  Ytest = sl.Y[folds$test]
  
  # build model and get prediction results
  output = sl.pred(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest, my.library = my.library)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end parallel processing
# stopCluster(cl)

# free memory
gc()

# combine the list of tables into one table
sl.cv = rbindlist(sl.cv)

# summarize performance metrics for every model
stat = c("Min", "Q1", "Median", "Mean", "Q3", "Max")

sl.diag = sl.cv[,.(stat = factor(stat, levels = stat),
                   ME = as.vector(summary(na.omit(ME))), 
                   RMSE = as.vector(summary(na.omit(RMSE))),
                   MAPE = as.vector(summary(na.omit(MAPE))),
                   Rsquared = as.vector(summary(na.omit(Rsquared))))]

# ---- Results ----------------------------------------------------------------------

# store model diagnostic results
sl.diag[, mod := rep("sl", nrow(sl.diag))]
mods.diag = rbind(mods.diag, sl.diag)

# store the model
sl.list = list("mod" = sl.mod)
mods.list$sl = sl.list

# shutdown the h2o instance if not done already
if("my.dnn" %in% my.library)
{
  h2o.shutdown(prompt = FALSE)
}

# remove objects we no longer need
rm(sl.pred, sl.cv, sl.X, sl.Y, sl.diag, sl.list, sl.mod, my.gbm, my.lm,
   my.pen, my.rf, my.dnn, my.svm, my.library)

# free memory
gc()

}

# -----------------------------------------------------------------------------------
# ---- Model Predictions ------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- Models ----------------------------------------------------------------------

{

# convert mods.diag into long format for plotting purposes
DT = data.table(melt(mods.diag, measure.vars = c("ME", "RMSE", "MAPE", "Rsquared")))

# convert mod into a factor for plotting purposes
DT[, mod := factor(mod, levels = unique(mod))]

# remove Inf values as these don't help
DT = data.table(DT[abs(value) < Inf])

# plot barplots of each diagnostic metric
diag.plot = ggplot(DT[stat == "Min" | stat == "Median" | stat == "Max"], aes(x = stat, y = value, group = reorder(paste0(mod, stat, variable), -value), fill = mod)) +
  geom_bar(stat = "identity", position = "dodge", color = "white") +
  scale_fill_manual(values = mycolors(length(levels(DT$mod)))) +
  labs(x = "Summary Statistic", y = "Value", fill = "Model") + 
  facet_wrap(~variable, scales = "free_y") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))

diag.plot

# remove objects we no longer need
rm(diag.plot, DT)

# free memory
gc()

}

# ---- Predictions -----------------------------------------------------------------

{

# ---- linear regression -----------------------------------------------------------

# build the model
set.seed(42)
mod = lm(SalePrice ~ ., data = train)

# make predictions with the training model using the test set
ynew = as.numeric(predict(mod, test))

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-lm.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew)

# ---- penalty regression ----------------------------------------------------------

# extract predictors (X), response (Y), and test set (newX)
X = as.matrix(train[,!"SalePrice"])
Y = train$SalePrice
newX = as.matrix(test)

# build the model
set.seed(42)
mod = cv.glmnet(x = X, y = Y, family = "gaussian", alpha = 0)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, s = mod$lambda.min, newX))

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-pen.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, X, Y, newX)

# ---- gradient boosting -----------------------------------------------------------

# extract predictors (X), response (Y), and test set (newX)
X = as.matrix(train[,!"SalePrice"])
Y = train$SalePrice
newX = as.matrix(test)

# build the model
set.seed(42)
mod = xgboost(label = Y, data = X,
              objective = "reg:linear", eval_metric = "rmse",
              eta = 0.1, max_depth = 4, nrounds = 1000, 
              min_child_weight = 3, gamma = 0, verbose = 0,
              subsample = 1, colsample_bytree = 1)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, newdata = newX))

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-gbm.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew, X, Y, newX)

# ---- random forest ---------------------------------------------------------------

# build the model
set.seed(42)
mod = randomForest(SalePrice ~ .,
                   data = train,
                   ntree = 1200,
                   nodesize = 9)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, newdata = test))

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-rf.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew)

# ---- deep neural network ---------------------------------------------------------

# initialize the h2o instance
h2o.init()
h2o.removeAll()

# remove the progress bar when model building
h2o.no_progress()

# make train and test into h2o objects
train.h2o = as.h2o(train)
test.h2o = as.h2o(test)

# identify predictors (x) and response (y)
y = "SalePrice"
x = names(test)

# build the training model
  # we are increasing the epochs because this can obnly improve the performance of our model
mod = h2o.deeplearning(y = y,
                       x = x,
                       training_frame = train.h2o,
                       hidden = c(334, 334, 334),
                       l1 = 1e-05,
                       l2 = 1e-05,
                       epochs = 1000,
                       seed = 42,
                       variable_importances = FALSE)

# make predictions with the training model using the test set
ynew = as.data.frame(predict(mod, newdata = test.h2o))$predict

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-dnn.csv", row.names = FALSE)

# shutdown the h2o instance
h2o.shutdown(prompt = FALSE)

# remove objects we no longer need
rm(mod, ynew, train.h2o, test.h2o, x, y)

# ---- support vector machine ------------------------------------------------------

# build the model
set.seed(42)
mod = svm(SalePrice ~ .,
          data = train,
          cost = 1.374917, 
          gamma = 1/39, 
          epsilon = 0.1)

# make predictions with the model using the test set
ynew = as.numeric(predict(mod, newdata = test))

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-svm.csv", row.names = FALSE)

# remove objects we no longer need
rm(mod, ynew)

# ---- super learner ---------------------------------------------------------------

# extract predictors (X), response (Y), and test set (newX)
sl.X = as.matrix(train[,!"SalePrice"])
sl.Y = train$SalePrice
sl.newX = as.matrix(test)

# create Super Learner wrappers

# linear regression wrapper
my.lm = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # build the training model
  set.seed(42)
  mod = lm(y ~ ., data = dat)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, data.table(newX)))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# penalty regression wrapper
my.pen = function(Y, X, newX, ...)
{
  # build the training model
  set.seed(42)
  mod = cv.glmnet(x = X, y = Y, family = "gaussian", alpha = 0)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, s = mod$lambda.min, newX))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# gradient boosting wrapper
my.gbm = function(Y, X, newX, ...)
{
  # build the training model
  set.seed(42)
  mod = xgboost(label = Y, data = X,
                objective = "reg:linear", eval_metric = "rmse",
                eta = 0.1, max_depth = 4, nrounds = 100, 
                min_child_weight = 3, gamma = 0, verbose = 0,
                subsample = 1, colsample_bytree = 1)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = newX))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# random forest wrapper
my.rf = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # build the training model
  set.seed(42)
  mod = randomForest(y ~ .,
                     data = dat,
                     ntree = 1200, 
                     nodesize = 9)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = data.table(newX)))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# deep neural network wrapper
my.dnn = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  dat[, y := Y]
  
  # make dat and newX into h2o objects
  dat.h2o = as.h2o(dat)
  newX.h2o = as.h2o(data.table(newX))
  
  # identify predictors (x) and response (y)
  y = "y"
  x = colnames(X)
  
  # build the training model
  mod = h2o.deeplearning(y = y,
                         x = x,
                         training_frame = dat.h2o,
                         hidden = c(334, 334, 334),
                         l1 = 1e-05,
                         l2 = 1e-05,
                         epochs = 100,
                         seed = 42,
                         variable_importances = FALSE)
  
  # make predictions with the training model using the test set
  ynew = as.data.frame(predict(mod, newdata = newX.h2o))$predict
  
  # run garbage collection to free up space for h2o
  gc()
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# support vector machine wrapper
my.svm = function(Y, X, newX, ...)
{
  # build the table for training the model
  dat = data.table(X)
  new.dat = data.table(newX)
  
  # give dat the response variable
  dat[, y := Y]
  
  # build the training model
  set.seed(42)
  mod = svm(y ~ .,
            data = dat,
            cost = 1.374917, 
            gamma = 1/39, 
            epsilon = 0.1)
  
  # make predictions with the training model using the test set
  ynew = as.numeric(predict(mod, newdata = new.dat))
  
  # return a list of the model (must label as fit) and predictions (must label as pred)
  output = list(pred = ynew, fit = mod)
  return(output)
}

# create a library of the above wrappers
my.library = list("my.gbm", "my.rf", "my.dnn", "my.svm")

# initialize the h2o instance if dnn is in sl
if("my.dnn" %in% my.library)
{
  h2o.init()
  h2o.removeAll()
  h2o.no_progress()
}

# build the model
set.seed(42)
mod = SuperLearner(Y = sl.Y, X = sl.X, newX = sl.newX, verbose = TRUE, 
                   family = gaussian(), SL.library = my.library)
mod

# make predictions with the model using the test set
ynew = as.numeric(mod$SL.predict)

# build submission
ynew = data.table(Id = (1:nrow(test)) + nrow(train),
                  SalePrice = ynew)

# export the submission
write.csv(ynew, file = "submission-nick-morris-sl.csv", row.names = FALSE)

# shutdown the h2o instance if dnn is in sl
if("my.dnn" %in% my.library)
{
  h2o.shutdown(prompt = FALSE)
}

# remove objects we no longer need
rm(mod, ynew, sl.X, sl.Y, sl.newX, my.lm, my.pen, my.gbm, my.rf, my.dnn, my.svm, my.library)

# free memory
gc()

}

}







