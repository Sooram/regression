setwd("C:/Users/Administrator/Documents/?��?��/")
rm(list=ls())

install.packages("mRMRe")
install.packages("car", repos="https://cran.rstudio.com/")
install.packages("leaps", repos="https://cran.rstudio.com/")


data <- read.csv("file_path",  header=TRUE)
str(data)
data <- subset(data,  select= -c('tags_to_be_removed'))

#--------------------------------------------------------------------------------------------------- mrmre
library(mRMRe)

data_mrmr <- mRMR.data(data = data)
filter <- mRMR.classic("mRMRe.Filter", data = data_mrmr, target_indices = 1, feature_count = ncol(data)-1)
sol <- solutions(filter) 
score <- scores(filter)

write.csv(score, "file_path")

col.names <- array(colnames(data))
ranked.feature <- array(sol$'1')
result <- data.frame()
for(i in ranked.feature) {
  print(col.names[i])
  result <- rbind(result, data.frame(col.names[i]))
}

write.csv(result, "file_path")

#--------------------------------------------------------------------------------------------------- regsubset
library(leaps)

# method: exhaustive, backward, forward, seqrep
regfit <- regsubsets(y ~ ., data = data, nvmax = ncol(data)-1, really.big=F, method="exhaustive")
reg.summary <- summary(regfit, all.best=TRUE, matrix=T, matrix.logical=T) 

plot(regfit)
# 
write.csv(reg.summary$which, "file_path")

# get the names of features when 'num' number of features are selected
get.features.of <- function(num, df.true.false) {
  colnames(df.true.false)[which(df.true.false[num,]==TRUE)]
}

get.features.of(12, reg.summary$which)

# decide number of features based on the value of Cp
plot(reg.summary$cp, xlab="Number of Features", ylab="Cp")
min.index <- which.min(reg.summary$cp)# the best number of features

points(min.index, reg.summary$cp[min.index], pch=20, col="red")

which.max(reg.summary$adjr2)
# get coef
coef(regfit, min.index)

#--------------------------------------------------------------------------------------------------- vif test
library(car)

fit <- lm(y~., data = data)
vif(fit)

