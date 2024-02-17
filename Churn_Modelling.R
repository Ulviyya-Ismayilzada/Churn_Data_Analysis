library(tidyverse)
library(h2o)
library(readr)
library(skimr)
install.packages('Information')
library(Information)
library(scorecard)
library(highcharter)
library(caret)
library(inspectdf)
library(glue)
library(ggsci)
library(yardstick)
library(dplyr)


#-----------------------------------Reviewing Dataset----------------------------------------------------

ch <- read.csv('Churn_Modelling (1).csv')
ch %>% view()


ch %>% glimpse()
ch %>% inspect_na()

ch$Exited %>% table() %>% prop.table
ch$Exited <-  ch$Exited %>% as.factor()


#--------------------------Removing unneeded columns,defining target and futures-----------------------------------------------------

iv <-ch %>% iv(y='Exited') %>% as.tibble() %>% 
    mutate(info_value=round(info_value,3)) %>% 
  arrange(desc(info_value))


infor_values <- iv %>% filter(info_value>0.02) %>% select(variable) %>% .[[1]]


ch_iv<- ch %>% select(Exited,infor_values)
ch_iv %>% view()
ch_iv %>% dim()
ch_iv<- ch_iv %>% select(-Surname)
bins <- ch_iv %>% woebin('Exited')
bins %>% view()
bins$NumOfProducts %>% woebin_plot()

ch_list <- ch_iv %>% split_df('Exited',ratio=0.8,seed=123)
train_woe <- ch_list$train %>% woebin_ply(bins)
test_woe <- ch_list$test %>% woebin_ply(bins)

names <- names(train_woe)
names <- gsub('_woe','',names)
names(train_woe ) <- names

names(test_woe) <- names


#--------------------------------Checking Multicollinearity--------------------------------------------------------
target <- 'Exited'
features <- train_woe %>% select(-'Exited') %>% names()



f <- as.formula(paste(target,paste(features,collapse = '+'),sep ='~'))
glm <- glm(f,data=train_woe,family='binomial')
summary(glm)

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features  %in% coef_na]

f <- as.formula(paste(target,paste(features,collapse = '+'),sep ='~'))
glm <- glm(f,data=train_woe,family='binomial')

while (glm %>% vif() %>% arrange(desc(gvif)) %>% .[[1,2]]>=3){
  aftervif <-  glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,'variable']
  aftervif <- aftervif$variable
  
  f <- as.formula(paste(target,paste(aftervif,collapse = '+'),sep="~"))
  glm <- glm(f, data=train_woe,family='binomial')
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features

#----------------------------Building churn model--------------------------------------------

h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()


model <- h2o.glm(x=features,y=target,family='binomial',
                 training_frame = train_h2o,validation_frame = test_h2o,
                 nfolds=5,compute_p_values = TRUE ,lambda = 0)

while (model@model$coefficients_table %>% 
       as.data.frame() %>%
       select(names,p_value) %>% 
       mutate(p_value=round(p_value),3) %>%
       .[-1,] %>% arrange(desc(p_value)) %>%
       .[1,2]>=0.05){
  model@model$coefficients_table %>% 
    as.data.frame() %>% select(names,p_value) %>% 
    mutate(p_value=round(p_value),3) %>% filter(!is.nan(p_value)) %>% .[-1,] %>% 
    arrange(desc(p_value)) %>% .[1,1] -> m
    
    features <- features[features!=m]


  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
    test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

    model <- h2o.glm(x=features,y=target,
                     family='binomial',
                 training_frame = train_h2o,
                 validation_frame = test_h2o,
                 nfolds=7,
                 compute_p_values = T,lambda = 0)  


}
model@model$coefficients_table %>% 
  as.data.frame() %>% 
  select(names,p_value) %>% 
  mutate(p_value=round(p_value,3))




model@model$coefficients %>% 
  as.data.frame() %>% 
  mutate(names = rownames(.)) %>%
  rename(coefficients = 1) %>%
  select( coefficients)


h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage!=0,] %>% 
  select(variable,percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'purple') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T) 

#-------------------Compare model results for training and test sets.----------------------------------

prediction <-model %>%
  h2o.predict(newdata=test_h2o) %>% 
  as.data.frame() %>% 
  select(p1,predict) %>% view()

model %>% h2o.performance(newdata=test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

prediction2 <-model %>%  
  h2o.predict(newdata=train_h2o) %>% 
  as.data.frame() %>% 
  select(p1,predict) %>% view()

model %>% h2o.performance(newdata=train_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')



#-------------------Evaluate and explain model results using ROC & AUC curves.------------------------

eval <- perf_eva(
  pred=prediction %>% pull(p1),
label=ch_list$test$Exited %>% 
  as.character() %>% as.numeric(),
binomial_metric = c('auc','gini'),show_plot = 'roc')

eval$binomial_metric$dat



#----------------------------------------Checking overfitting--------------------------------------------------------------------------------

model %>% h2o.auc(train = T,valid = T,xval=T) %>%
  as.tibble() %>% round(2) %>%
  mutate(data=c('train','test','cross_val')) %>%
  mutate(gini=2*value-1) %>% select(data,auc=value,gini)



