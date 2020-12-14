# Credit Card Fraud

# read the dataset
credit_card <- read.csv('creditcard.csv')

# view the structure of dataset
str(credit_card)

# convert Class to vector variable
credit_card$Class <- factor(credit_card$Class, levels=c(0, 1))

# get the summary of data
summary(credit_card)

# find missing values
sum(is.na(credit_card))

# obtain the distribution of real & fraud transactions
table(credit_card$Class)

# obtain the percentage of real & fraud transactions
prop.table(table(credit_card$Class))

# data visualization
labels <- c('legitimate', 'fraud')
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)), 2))
labels <- paste0(labels, '%')
labels
pie(table(credit_card$Class), labels, col = c('green', 'red'),
    main = 'Credit Card Transaction')


# model prediction

predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0, 1))


library(ggplot2)
library(lattice)
library(caret)

confusionMatrix(data=predictions, reference=credit_card$Class)


#install.packages('dplyr')
library(dplyr)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)


library(ggplot2)
ggplot(data = credit_card, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))


# split the dataset into training and test set

install.packages('caTools')
library(caTools)

set.seed(123)

data_sample <- sample.split(credit_card$Class, SplitRatio = 0.80)

train_data <- subset(credit_card, data_sample==TRUE)

test_data <- subset(credit_card, data_sample==FALSE)

dim(train_data)
dim(test_data)


# Random Over-Sampling (ROS)

table(train_data$Class)

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit 


#install.packages('ROSE')
library(ROSE)

oversampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = 'over',
                                   N = new_n_total,
                                   seed = 2019)

oversampled_credit <- oversampling_result$data

table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.2)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))


# Random Under_Sampling (RUS)

table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud


undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = 'under',
                                    N = new_n_total,
                                    seed = 2019)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)
  
ggplot(data = undersampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
  


# Combine both methods ROS and RUS

n_new <- nrow(train_data)
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~ .,
                               data = train_data,
                               method = 'both',
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)

prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.2)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
  

# balance the dataset

install.packages('smotefamily')
library(smotefamily)  

table(train_data$Class)


# set the number of good and bad cases and the desired percent of good cases

n0 <- 22750
n1 <- 35
r0 <- 0.6


# calculate the value for the dup_size parameter

ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1


smote_output <- SMOTE(X = train_data[ , -c(1, 31)],
                      target = train_data$Class,
                      K = 5,
                      dup_size = ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- 'Class'

prop.table(table(credit_smote$Class))

# class distribution for the original dataset using 

ggplot(train_data, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

# class distribution for over-sampling dataset using SMOTE

ggplot(credit_smote, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
  

#install.packages('rpart')
#install.packages('rpart.plot')

library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ . , credit_smote)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# predict fraudulent transactions
predicted_val <- predict(CART_model, test_data, type='class')

# confusion matrix

library(caret)

confusionMatrix(predicted_val, test_data$Class)

predicted_val <- predict(CART_model, credit_card[-1], type = 'class')
confusionMatrix(predicted_val, credit_card$Class)

# decision tree without SMOTE
CART_model <- rpart(Class ~ . , train_data[,-1])

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# predict frauds

predicted_val <- predict(CART_model, test_data[-1], type = 'class')

confusionMatrix(predicted_val, test_data$Class)

predicted_val <- predict(CART_model, credit_card[-1], type = 'class')

confusionMatrix(predicted_val, credit_card$Class)










