
## Partitioning Data into Train and Test datasets in 70:30
library(caret)
set.seed(1234567)
train1<-createDataPartition(bank_full$subscribed,p=0.7,list=FALSE)
train<-bank_full[train1,]
test<-bank_full[-train1,]

## CLASSIFICATION USING NEURAL NETWORK

library(nnet)

## 1. Fit a Single Hidden Layer Neural Network using Least Squares
train.nnet<-nnet(subscribed~.,train,size=3,rang=0.07,Hess=FALSE,decay=15e-4,maxit=250)
## Use TEST data for testing the trained model
test.nnet<-predict(train.nnet,test,type=("class"))
## MisClassification Confusion Matrix
table(test$subscribed,test.nnet)
## One can maximize the Accuracy by changing the "size" while training the neural network. SIZE refers to the number of nodes in the hidden layer.
which.is.max(test.nnet)  ## To Fine which row break ties at random (Maximum position in vector)

##2. Use Multinomial Log Linear models using Neural Networks
train.mlln<-multinom(subscribed~.,train)
##USe TEST data for testing the trained model
test.mlln<-predict(train.mlln,test)
##Misclassification or Confusion Matrix
table(test$subscribed,test.mlln)
##3. Training Neural Network Using BACK PROPOGATION
install.packages("neuralnet")
library(neuralnet)
## Check for all Input Independent Variables to be Integer or Numeric or complex matrix or vector arguments. If they are not any one of these, then tranform them accordingly
str(train)
str(test)
## It can be observed that all are either integer or factor. Now these factors have to be transformed to numeric.
## One cannot use directly as.numeric() to convert factors to numeric as it has limitations.
## First, Lets convert factors having character levels to numeric levels
str(bank_full)
bank_full_transform<-bank_full
bank_full_transform$marital=factor(bank_full_transform$marital,levels=c("single","married","divorced"),labels=c(1,2,3))
str(bank_full_transform)
## Now convert these numerical factors into numeric
bank_full_transform$subscribed<-as.numeric(as.character(bank_full_transform$subscribed))

str(bank_full_transform)
## Now all the variables are wither intergers or numeric
## Now we shall partition the data into train and test data
library(caret)
set.seed(1234567)
train2<-createDataPartition(bank_full_transform$subscribed,p=0.7,list=FALSE)
trainnew<-bank_full_transform[train2,]
testnew<-bank_full_transform[-train2,]
str(trainnew)
str(testnew)
## Now lets run the neuralnet model on Train dataset
trainnew.nnbp<-neuralnet(subscribed~age+balance+lastday+lastduration+numcontacts+pdays+pcontacts+marital+education+housingloan+personalloan+lastmonth+poutcome+lastcommtype,data=bank_full_transform,hidden=5,threshold=0.01,err.fct="sse",linear.output=FALSE,likelihood=TRUE,stepmax=1e+05,rep=1,startweights=NULL,learningrate.limit=list(0.1,1.5),learningrate.factor=list(minus=0.5,plus=1.5),learningrate=0.5,lifesign="minimal",lifesign.step=1000,algorithm="backprop",act.fct="logistic",exclude=NULL,constant.weights=NULL)
## Here, Back Propogation Algorithm has been used. One can also use rprop+ (resilient BP with weight backtracking),rprop- (resilient BP without weight backtracking), "sag and "slr" as modified global convergent algorithm
## Accordingly the accuracy can be checked for each algorithm
summary(train.nnbp)
## Plot method for the genralised weights wrt specific covariate (independent variable) and response target variable
gwplot(trainnew.nnbp,selected.covariate="balance")
##(Smoother the Curve- Better is the model prediction)
## Plot the trained Neural Network
plot(trainnew.nnbp,rep="best")
## To check your prediction accuracy of training model
prediction(trainnew.nnbp)
print(trainnew.nnbp)
## Now use the TEST data set to test the trained model
## Make sure that target column in removed from the test data set
columns=c("age","job","marital","education","balance","housingloan","personalloan","lastcommtype","lastday","lastmonth","lastduration","numcontacts","pdays","poutcome")
testnew2<-subset(testnew,select=columns)
testnew.nnbp<-compute(trainnew.nnbp,testnew2,rep=1)
## MisClassification Confusion Matrix
table(testnew$subscribed,testnew.nnbp$net.result)
cbind(testnew$subscribed,testnew.nnbp$net.result)
print(testnew.nnbp)
