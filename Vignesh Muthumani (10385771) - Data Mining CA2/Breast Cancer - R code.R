# Vignesh Muthumani (10385771) - Data Mining CA2 - R code

# reading the data
bc <- read.csv(file.choose(), header=T, stringsAsFactors=F)

# viewing the data
View(bc)

# initial data exploration
summary(bc)
str(bc)

# checking for missing values
is.na(bc) # we can see that there aren't any missing values in our dataset
bc[!complete.cases(bc),]

# removing the last column 'x' by making it NULL
bc$X <- NULL

# removing the first column 'id' as it's not necessary 
bc <- bc[,-1]

# converting the chr variable into a factor variable
class(bc$diagnosis)
str(bc$diagnosis)
bc$diagnosis <- factor(ifelse(bc$diagnosis=="B", "Benign", "Malignant"))
class(bc$diagnosis)

par(mfrow=c(1,1))
boxplot(bc)

# data exploration - after refining the dataset
str(bc)
summary(bc) 
View(bc)
dim(bc)

# feature density
library(caret)

scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.6)

featurePlot(x=bc[,-1], y=bc$diagnosis, plot="density",scales=scales,
            layout = c(3,10), auto.key = list(columns = 2), pch = "|")

# pearson correlation analysis
nc=ncol(bc)
df <- bc[,1:nc]
df$diagnosis <- as.integer(factor(df$diagnosis))-1
correlations <- cor(df,method="pearson")
library(corrplot)
corrplot(correlations, number.cex = .9, method = "square", 
         hclust.method = "ward", order = "FPC",
         type = "full", tl.cex=0.8,tl.col = "black")

# Principal component analysis
bc.pca <- prcomp(bc[-1], center=TRUE, scale.=TRUE)
summary(bc.pca)
plot(bc.pca, type="l", main='')
grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()
biplot(bc.pca)

# splitting the dataset for training the model
set.seed(218)	# for the consistency

nrows <- nrow(bc)
index <- sample(1:nrows, 0.7 * nrows)	# shuffle and divide

#splitting the dataset
train <- bc[index,]			        
test <- bc[-index,]  		        

# checking the dimensions after the split
dim(bc)    # 569 rows (100%)
dim(train) # 398 rows (70%)
dim(test)  # 171 rows (30%)

prop.table(table(train$diagnosis)) # proportion of diagnosis in train set
prop.table(table(test$diagnosis)) # proportion of diagnosis in test set

# Data modelling

# (1) Naive Bayes: 
library(e1071)
library(caret)

# fitting the model
naiveb <- naiveBayes(train[,-1], train$diagnosis)

# predicting test set with the trained model
predicted_nb <- predict(naiveb, test[,-1])

# confusion matrix to see the performance
con_matrix_nb <- confusionMatrix(predicted_nb, test$diagnosis)		
con_matrix_nb

# (2) SVM:

# fitting the model
svm <- svm(diagnosis~., data=train)

# predicting test set with the trained model
predicted_svm <- predict(svm, test[,-1])

# confusion matrix to see the performance
con_matrix_svm <- confusionMatrix(predicted_svm, test$diagnosis)
con_matrix_svm

# (3) Random Forest:
library(randomForest)

# fitting the model
ran_forest <- randomForest(diagnosis~., data=train, ntree=500, proximity=T, importance=T)

# performance of random forest model
plot(ran_forest, main="Random Forest: MSE error vs. no of trees")

# predicting test set with the trained model
predicted_rf   <- predict(ran_forest, test[,-1])

# confusion matrix to see the performance
con_matrix_rf    <- confusionMatrix(predicted_rf, test$diagnosis)
con_matrix_rf

# comparison of all the models to assess their performances
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(1,3))
fourfoldplot(con_matrix_nb$table, color = col, conf.level = 0, margin = 1, main=paste("NaiveBayes (",round(con_matrix_nb$overall[1]*100),"%)",sep=""))
fourfoldplot(con_matrix_svm$table, color = col, conf.level = 0, margin = 1, main=paste("SVM (",round(con_matrix_svm$overall[1]*100),"%)",sep=""))
fourfoldplot(con_matrix_rf$table, color = col, conf.level = 0, margin = 1, main=paste("RandomForest (",round(con_matrix_rf$overall[1]*100),"%)",sep=""))

# selecting the best model
opt_prediction <- c( con_matrix_nb$overall[1], con_matrix_svm$overall[1], con_matrix_rf$overall[1])
names(opt_prediction) <- c("Naive Bayes","SVM","Random Forest")
best_model <- subset(opt_prediction, opt_prediction==max(opt_prediction))
best_model

# testing the best model for prediction

new_test <- read.csv(file.choose(), header=T, stringsAsFactors=F)
new_test$X <- NULL

#malignent patient
A <- new_test[35,]   	    	## 35th patient
A[,c(1,2)]

#benign patient 
B <- new_test[56,]              	## 56th patient          
B[,c(1,2)]

#delete diagnosis column for testing
A$diagnosis <- NULL
B$diagnosis <- NULL

# patient cancer diagnosis prediction function
# for printing output
cancer_prediction <- function(new, method=svm) {
  new_prediction <- predict(method, new[,-1])
  new_result <- as.character(new_prediction)
  return(paste("Patient ID: ",new[,1],"  =>  Result: ", new_result, sep=""))
}

cancer_prediction(A)
cancer_prediction(B)

# K-fold cross validation for svm with 10 folds
ControlParameters <- trainControl(method = 'repeatedcv',
                                  number = 10, repeats = 5,
                                  savePredictions = TRUE,
                                  classProbs = TRUE)


modelsvm <- train(diagnosis~., data=bc, method = "svmLinear",
                  trControl = ControlParameters)

modelsvm

tune.svm(diagnosis~., data= bc) #gives error estimation

######################################################################
