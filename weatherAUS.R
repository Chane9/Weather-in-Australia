library(tidyr)
library(caret)
library(dplyr)
library(MASS)
library(ISLR)
library(randomForest)
library(tree)
library(rpart)
library(adabag)
############################ FEATURE SELECTION #####################################
weatherAUS <- read.csv('weatherAUS.csv', header = TRUE, stringsAsFactors = TRUE)
# first thing, remove RISK_MM
weatherAUS <- weatherAUS[,-c(23)]
# convert date data
weatherAUS <- separate(weatherAUS, "Date", c("Year", "Month", "Day"), sep = "-")
weatherAUS$Year <- as.numeric(weatherAUS$Year)
weatherAUS$Month <- as.numeric(weatherAUS$Month)
weatherAUS$Day <- as.numeric(weatherAUS$Day)
weatherAUS_withoutNA <- weatherAUS[complete.cases(weatherAUS),]

# first check whether the dataset without NA has different percentage with that of dataset with NA
summary(weatherAUS_withoutNA$RainTomorrow)
12427/(12427+43993) # 0.2203
summary(weatherAUS$RainTomorrow)
31877/(31877+110316) # 0.2242
# around the same
######## check correlation for numeric data
# first seperate date variable and transform
weatherAUS_num <- weatherAUS_withoutNA[,-c(4, 10, 12, 13, 24)]
correlationMatrix <- cor(weatherAUS_num[,1:19])
findCorrelation = findCorrelation(correlationMatrix, cutoff=0.75) # put any value as a "cutoff"
objects(weatherAUS_num[,findCorrelation])
# discard highly correlated variables
weatherAUS <- subset(weatherAUS, select = -c(MaxTemp, Pressure3pm, Temp3pm, Temp9am))

######## check categorical data: use chi-sq test 
weatherAUS_categorical <- names(which(sapply(weatherAUS, class) == "factor"))
weatherAUS_categorical <- setdiff(weatherAUS_categorical, "RainTomorrow")
chisq_test <- lapply(weatherAUS_categorical, function(x) { 
  chisq.test(weatherAUS[,x], weatherAUS[, "RainTomorrow"], simulate.p.value = TRUE)
})
chisq_test
# retain all categorical data 

######### DROP VARIABLE CONTAINING HIGH PERCENTAGE OF MISSING VALUE
summary(weatherAUS$Evaporation)
67816/142193 # missing value for sunshine is high to 47.7%
summary(weatherAUS$Sunshine)
60843/142193 # both because Evaporation has outliers and around 42.7% missing value in total thus drop it
summary(weatherAUS$Cloud3pm)
57094/142193 # around 40% thus drop it.
weatherAUS <- subset(weatherAUS, select = -c(Sunshine, Evaporation, Cloud3pm))
str(weatherAUS)


############################### MISSING VALUE ####################################
# categorical data 
colSums(is.na(weatherAUS))
missing_categorical_col <- c("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")
for (i in missing_categorical_col){
  weatherAUS[,i][is.na(weatherAUS[,i])] <- names(sort(table(weatherAUS[,i]), decreasing=TRUE)[1])
}
# numeric data, fillin with median
numeric_col <- c("MinTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
                 "Humidity9am", "Humidity3pm", "Pressure9am", "Cloud9am")
for (i in numeric_col){
  weatherAUS[,i][is.na(weatherAUS[,i])] <- median(weatherAUS[,i], na.rm=TRUE)
}
colSums(is.na(weatherAUS))

############################## TRAIN TEST DATASET SPLIT ###########################
# set the seed to make partition reproducible
set.seed(123)
sample_size <- floor(0.8*nrow(weatherAUS))
train_ind <- sample(seq_len(nrow(weatherAUS)), size = sample_size)
train <- weatherAUS[train_ind, ]
test <- weatherAUS[-train_ind, ]
Direction_test <- test$RainTomorrow # 1 No 2 Yes
# drop Date first
str(train)
glm.fit <- glm(RainTomorrow ~ ., data = weatherAUS, family = binomial)
# further improve the data choosen: remove Year and Day
summary(glm.fit)
weatherAUS <- subset(weatherAUS, select = -c(Year, Day))
str(weatherAUS)
# train and test dataset update
train <- weatherAUS[train_ind, ]
test <- weatherAUS[-train_ind, ]
Direction_test <- test$RainTomorrow # 1 No 2 Yes


############################### Model Application #################################
######### LOGISTIC REGRESSION 
LogReg.model <- glm(RainTomorrow ~ . , data = train, family = binomial)
LogReg_probs = predict(LogReg.model, test, type = "response")
dim(test)
LogReg.model
LogReg_pred = rep("No", 28439)
LogReg_pred[LogReg_probs > .5] = "Yes"
table(LogReg_pred, Direction_test)
# Proportation of make correct classification
mean(LogReg_pred == Direction_test)  # 0.8439115
# Misclassfication error rate
mean(LogReg_pred != Direction_test)  # 0.1560885
# confusion matrix
LogReg_pred2 <- as.factor(LogReg_pred)
confusionMatrix(LogReg_pred2, Direction_test)

######## Classification Tree
tree.model <- tree(RainTomorrow ~ . - Location, data = train) # factor predictors must have at most 32 levels
summary(tree.model)
tree.model
plot(tree.model)
text(tree.model, pretty = 0, cex = 0.8)
tree.pred <- predict(tree.model, test, type = 'class')
# Proportation of make correct classification
mean(tree.pred == Direction_test)   # 0.8225324
## Mis-classification error
mean(tree.pred != Direction_test)  # 0.1774676
## Confusion matrix
treetab <- table(tree.pred, Direction_test)
treeresult <- confusionMatrix(treetab)
treeresult
precision <- treeresult$byClass['Pos Pred Value']
precision # 0.8262467
recall <- treeresult$byClass['Sensitivity']
recall  # 0.9770547
# cross-validation to find the best tree
set.seed(10)
cv_tree <- cv.tree(tree.model, FUN = prune.misclass)
cv_tree
par(mfrow = c(1,2))
plot(cv_tree$size, cv_tree$dev, type='b')
plot(cv_tree$k, cv_tree$dev, type='b')
prune.weather <- prune.misclass(tree.model, best = 3)
plot(prune.weather)
text(prune.weather, pretty = 0)
prune.weather
prune.pred <- predict(prune.weather, test, type = 'class')
mean(prune.pred == Direction_test)   # 0.8225324
## Mis-classification error
mean(prune.pred != Direction_test)  # 0.1774676
prunetab <- table(prune.pred, Direction_test)
pruneresult <- confusionMatrix(prunetab)
pruneresult
precision <- pruneresult$byClass['Pos Pred Value']
precision # 0.8262467
recall <- pruneresult$byClass['Sensitivity']
recall  # 0.9770547

######## LDA
lda.model <- lda(RainTomorrow ~ ., data =train, family = binomial)
names(predict(lda.model, test))
lda.pred.posterior = predict(lda.model,test)$posterior
head(lda.pred.posterior)
lda.pred = predict(lda.model, test)$class
head(lda.pred)
table(lda.pred, Direction_test)
# Proportion of make correct classification
mean(lda.pred == Direction_test)  # 0.8436302
# Mis-classification error
mean(lda.pred != Direction_test)  # 0.1563698
ldatab <- table(lda.pred, Direction_test)
ldaresult <- confusionMatrix(ldatab)
ldaresult
precision <- ldaresult$byClass['Pos Pred Value']
precision # 0.8262467
recall <- ldaresult$byClass['Sensitivity']
recall  # 0.9770547
lda.model

######## QDA
qda.model <- qda(RainTomorrow ~ ., data = train, family=binomial)
str(qda.model)
qda.pred = predict(qda.model, test)$class
table(qda.pred, Direction_test)
# Proportion of make correct classification
mean(qda.pred == Direction_test)  # 0.6506206
# Mis-classification error
mean(qda.pred != Direction_test)  # 0.3493794
qdatab <- table(qda.pred, Direction_test)
qdaresult <- confusionMatrix(qdatab)
qdaresult
precision <- qdaresult$byClass['Pos Pred Value']
precision # 0.8262467
recall <- qdaresult$byClass['Sensitivity']
recall  # 0.9770547
# library(klaR)
# partimat(RainTomorrow ~ RainToday, data = train, method = "qda", plot.matrix = TRUE, col.correct='green', col.wrong='red')

######## SVM
svmfit <- svm(RainTomorrow ~ ., data = train, kernel = "linear", cost = 10, scale = FALSE)
svmfit$index # took too long to run computer crushed
svm.pred <- predict(svmfit, newdata = test)
# Proportion of make correct classification
mean(svm.pred == Direction_test)  # 0.8414853
# Mis-classification error
mean(svm.pred != Direction_test)  # 0.1585147
# Precison
# set.seed(1)
# tune.out = tune(svm, RainTomorrow ~ ., data = train, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# summary(tune.out)

######### Bagging
set.seed(1)
Bag_model <- randomForest(RainTomorrow~., data = train, mtry=15,
                           importance=TRUE, ntree = 500)
Bag_model
Bag.pred <- predict(Bag_model, newdata = test)
confusionMatrix(Bag.pred, Direction_test)
par(mfrow = c(1,1))
plot(Bag_model)
importance(Bag_model)
varImpPlot(Bag_model)
table(Bag.pred, Direction_test)
# Proportion of make correct classification
mean(Bag.pred == Direction_test)  # 0.8540736
# Mis-classification error
mean(Bag.pred != Direction_test)  # 0.1459264
# node impurity in classification is Gini. 
# Humifity3pm is the most important one
Bagresult <- confusionMatrix(Bag.pred, Direction_test)
Bagresult
precision <- Bagresult$byClass['Pos Pred Value']
precision #  portion of the true positive examples in positively classified examples
recall <- Bagresult$byClass['Sensitivity']
recall  # portion of correctly classified positive examples

######### Random Forest
RF_model <- randomForest(RainTomorrow~., data = train, mtry = 4,
                          importance=TRUE, ntree = 1000)
RF_model
varImpPlot(RF_model)
RF.pred <- predict(RF_model, newdata = test)
table(RF.pred, Direction_test)
# Proportion of make correct classification
mean(RF.pred == Direction_test)  # 0.8566054 accuracy
# Mis-classification error
mean(RF.pred != Direction_test)  # 0.1433946
confusionMatrix(RF.pred, Direction_test)
precision <- treeresult$byClass['Pos Pred Value']
precision # 
recall <- treeresult$byClass['Sensitivity']
recall

###### adaboost
ada_model <- boosting(RainTomorrow~., data=train, boos=TRUE, mfinal=100)
ada_model$trees[1]
ada.pred <- predict.boosting(ada_model, test)
ada.pred$confusion
table(ada.pred$class, Direction_test)
mean(ada.pred$class == Direction_test)  # 0.8491508 accuracy
# Mis-classification error
mean(ada.pred$class != Direction_test)  # 0.1508492 misclassification error
ada.pred1 <- as.factor(ada.pred$class)
confusionMatrix(ada.pred1, Direction_test)
errorevol(ada_model, weatherAUS)

