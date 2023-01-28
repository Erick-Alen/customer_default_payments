---
title: "R Notebook"
output: html_notebook
---
# Installing all project packages

#Checking null attributes
install.packages("Amelia")

#Preprocessing data with ML
install.packages("caret")

#Plotting graphics
install.packages("ggplot2")

#Manipulating data
install.packages("dplyr")

#Reshape data
install.packages("reshape")

#Training model
install.packages("randomForest")
install.packages("e1071")


```{r}
# Loading packages
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)
```


```{r}
# Loading dataset
# site: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dados_clientes <- read.csv("dados/dataset.csv")


```



```{r}
# Data and structure visualizing
head(dados_clientes)
dim(dados_clientes)
str(dados_clientes) 
summary(dados_clientes)
```
#################### Exploratory Analysis, Cleaning and Transformation ####################

```{r}
# Removing 'ID' Column
dados_clientes$ID <- NULL
dim(dados_clientes)
head(dados_clientes)
```

```{r}
# Renaming class column
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
head(dados_clientes)
```

```{r}
# Removing absent values from dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Missing Values")
dados_clientes <- na.omit(dados_clientes)
```

```{r}
# Converting integer attributes to factors (type and values)
str(dados_clientes)
```

```{r}
# SEX
head(dados_clientes$SEX) 
str(dados_clientes$SEX) 
summary(dados_clientes$SEX) 
dados_clientes$SEX <- cut(dados_clientes$SEX, 
                             c(0,1,2), 
                             labels = c("Male",
                                        "Female"))
head(dados_clientes$SEX) 
str(dados_clientes$SEX) 
summary(dados_clientes$SEX) 
```

```{r}
# EDUCATION
str(dados_clientes$EDUCATION)
summary(dados_clientes$EDUCATION) 
dados_clientes$EDUCATION <- cut(dados_clientes$EDUCATION, 
                                c(0,1,2,3,4),
                                   labels = c("Post graduate",
                                              "Graduate",
                                              "College",
                                              "Others"))
head(dados_clientes$EDUCATION) 
str(dados_clientes$EDUCATION) 
summary(dados_clientes$EDUCATION) 
```

```{r}
# MARRIAGE
str(dados_clientes$MARRIAGE) 
summary(dados_clientes$MARRIAGE) 
dados_clientes$MARRIAGE <- cut(dados_clientes$MARRIAGE, 
                                   c(-1,0,1,2,3),
                                   labels = c("Unknown",
                                              "Married",
                                              "Single",
                                              "Others"))
head(dados_clientes$MARRIAGE) 
str(dados_clientes$MARRIAGE) 
summary(dados_clientes$MARRIAGE) 
```

```{r}
# Grouping clients by age
str(dados_clientes$AGE) 
summary(dados_clientes$AGE) 
hist(dados_clientes$AGE)

dados_clientes$AGE <- cut(dados_clientes$AGE, 
                            c(0,30,50,100), 
                            labels = c("Young", 
                                       "Adult", 
                                       "Old"))
head(dados_clientes$AGE) 
str(dados_clientes$AGE) 
summary(dados_clientes$AGE)
head(dados_clientes)
```

```{r}
# Converting payment variables to factor type only
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Dataset after manipulation 
str(dados_clientes) 
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)
head(dados_clientes)
```

```{r}
# Changing 'inadimplente' to factor type
str(dados_clientes$inadimplente)
colnames(dados_clientes)
dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente)
str(dados_clientes$inadimplente)
head(dados_clientes)
```

```{r}
# 'delinquents' versus 'non-delinquents'
table(dados_clientes$inadimplente)

```

```{r}
#Proportion between classes
prop.table(table(dados_clientes$inadimplente))
```

```{r}
# Distribution plot using ggplot2
qplot(inadimplente, data = dados_clientes, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```

```{r}
# Set seed
set.seed(12345)

# stratified samples using ggplot2. 
# Select 75 percent of the lines in 'inadimplente' to another partition
# creating a dataset to train the data
indice <- createDataPartition(dados_clientes$inadimplente, p = 0.75, list = FALSE)
dim(indice)

dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$inadimplente)
```

```{r}
# Percentages between classes
prop.table(table(dados_treino$inadimplente))
# Register numbers in training dataset
dim(dados_treino)
```

```{r}
# Comparing proportions between training data and original data
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)), 
                       prop.table(table(dados_clientes$inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados
```

```{r}
# Melt Data - Convert column in lines
melt_compara_dados <- melt(compara_dados)
melt_compara_dados
```

```{r}
# Plot training distribution vs original
ggplot(melt_compara_dados, aes(x = X1, y = value)) + 
  geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```

#################### ML Model ####################

```{r}
# creating the test data, which excludes all variables in training
dados_teste <- dados_clientes[-indice,]
dim(indice)
dim(dados_teste)
dim(dados_treino)
```

```{r}
# building first model version with random forest(imbalanced data)
head(dados_treino)
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1
```

```{r}
# Evaluating model
plot(modelo_v1)
```

```{r}
# Predictions with test data
previsoes_v1 <- predict(modelo_v1, dados_teste)
```

```{r}
# Confusion Matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1
```

```{r}
#Precision, Recall e F1-Score and  model evaluation metrics
y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision
```

```{r}
recall <- sensitivity(y_pred_v1, y)
recall
```

```{r}
F1 <- (2 * precision * recall) / (precision + recall)
F1
```

