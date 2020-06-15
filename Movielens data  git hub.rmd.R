
---
  title: "Capstone Project Movielens"
author: "Kanchan Singh"
output:
  pdf_document:
  latex_engine: xelatex
fontsize: 12pt
---
  
  **Introduction**
  
  This project is part of the capstone project of the EdX course ‘HarvardX: PH125.9x Data Science: Capstone’. In this project, I have used R programming skill that I learned during the HarvardX’s Data Science Professional Certificate Program . In this project, I have analyzed a dataset called ‘Movielens’, which contains millions of movie ratings by users. I have used insights from this analysis to generate predictions of movies, comparing with the actual ratings of movies to check the quality of the prediction algorithm. I have also developed a recommendation system for recommending movies.

Recommendation systems plays an important role in e-commerce and online streaming services by increasing user retention and satisfaction, leading to sales and profit growth. Recommendation system also saves users’ searching time by providing quality feedback from other customers. Usually recommendation systems are based on rating system 1 to 5, 1 for lowest rating and 5 for highest rating.

In this document, I have loaded the dataset, summarized the goal of the project and explained the process and techniques used such as data cleaning, data exploration and visualization. Based on insights gained, I used the modeling approach. I have 
concluded this report by brief summary of findings, its limitations and future work.

Model Evaluation
I compared the predicted value with the actual outcome to evaluate the machine learning algorithms by using loss function that measures the difference between both values. The most common loss functions in machine learning are the mean absolute error (MAE), mean squared error (MSE) and root mean squared error (RMSE).
Regardless of the loss function, when the user consistently selects the predicted movie, the error is equal to zero and the algorithm is perfect. The goal of this project is to create a recommendation system with RMSE lower than 0.8649. There’s no specific target for MSE and MAE.



#Important Library
```{r,warning=FALSE,message=FALSE}
library(tidyr)
library(pdftools)
library(knitr)
library(modelr)
library(dslabs)
library(formatR)
library(tinytex)
library(data.table)
library(stringr)
library(dplyr)
library(ggplot2)
library(lattice)
library(tidyr)
library(broom)
library(caret)
library(ggthemes)
library(scales)
library(lubridate)
library(yaml)

```

**Data Preparation**
  
  *In this section, I have downloaded and prepared the dataset to be used in the analysis. I split the dataset in two parts, the training set called edx and the evaluation set called validation with 90% and 10% of the original dataset respectively.*
  
  
  ```{r,results='hide',warning=FALSE}
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)



rm(dl, ratings, movies, test_index, temp, movielens, removed)

```


*Now, I split the edx set in two parts, the train set and test set with 90% and 10% of edx set respectively. The model is created and trained in the train set and tested in the test set until the RMSE target is achieved, then finally I train the model again in the entire edx set and validate in the validation set. The name of this method is cross-validation.*
  
  
  ```{r,warning=FALSE}
#Breaking edx into two parts
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

```

*In this part , I did some exploratory analysis because this helps to create better model,exploring the structure of the data, the distribution of ratings and the relationship of the predictors.*
  
  **#how many rows and columns in edx**
  
  ```{r}
nrow(edx)
ncol(edx)

```

**How many zeros and threes were given in the edx dataset?**
  
  ```{r}
sum(edx$rating == 0)

sum(edx$rating == 3)

```

**How many are movies in edx?**
  
  ```{r}

edx%>% summarise(n_movies=n_distinct(movieId))
```

**How many are users in edx?**
  
  ```{r}
edx%>% summarise(n_users=n_distinct(userId))
```

**Which movie has the greatest number of ratings?**
  
  ```{r}
edx %>% group_by(title) %>% summarise(number = n()) %>% arrange(desc(number))

```

**# How many are different movies in the data set?**
  
  ```{r}
n_distinct(movielens$genres)
```

**Convert Timestamp to year**
  ```{r}
edx <- mutate(edx, year_rated = year(as_datetime(timestamp)))
head(edx)
```

**Visualization of Data**
  
  *In this part, I explored some plots for some distribution such as rating, user and movie .*
  
  **Rating distribution per year**
  
  ```{r}
edx %>% mutate(year = year(as_datetime(timestamp, origin="1970-01-01"))) %>%
  ggplot(aes(x=year)) +
  geom_histogram(color = "white") + 
  ggtitle("Rating Distribution Per Year") +
  xlab("Year") +
  ylab("Number of Ratings") 
```
```
**Movie distribution**
  
  ```{r,echo=FALSE}
edx %>% group_by(movieId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + geom_histogram(fill = "cadetblue3", color = "grey20", bins = 10) +
  scale_x_log10() +
  ggtitle("Number of Movies Ratings")
```

**User distribution**
  
  ```{r}
edx %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + geom_histogram(fill = "red", color = "grey20", bins = 10) +
  scale_x_log10() +
  ggtitle("Number of users' Ratings")

```

**Movie rating averages**
  
  ```{r}
movie_avgs <- edx %>% group_by(movieId) %>% summarize(avg_movie_rating = mean(rating))
user_avgs <- edx %>% group_by(userId) %>% summarize(avg_user_rating = mean(rating))
year_avgs <- edx%>% group_by(year_rated) %>% summarize(avg_rating_by_year = mean(rating))
head(movie_avgs)
head(user_avgs)
head(year_avgs)
```

**user avarage distribution**
  
  ```{r}
user_avgs %>%
  ggplot(aes(userId, avg_user_rating)) +
  geom_point(alpha=0.2,colour="purple") +
  ggtitle("User vs Average User Rating")
```

**Model Exploration**
  
  **The most basic model is loss functions model.In this analysis , I used Root Mean Squared Error. **
  
  ```{r}
# Define mean absolute error
MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}

# Define Mean Squared Error (MSE)
MSE <- function(true_ratings, predicted_ratings){
  mean((true_ratings - predicted_ratings)^2)
}

# Define Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
```

**Random Prediction**
  
  *Random Prediction model randomly predicts the ratings using the observed probabilities in the training set.The probability of each rating in the training set is calculated, then the rating for the test set is predicted and actual rating. is compared with predicted rating. *
  
  *Since the training set is a sample of the entire population and the real distribution of ratings is unknown, the Monte Carlo simulation with replacement provides a good approximation of the rating distribution.*
  
  ```{r,warning=FALSE}
set.seed(1, sample.kind = "Rounding")

# Create the probability of each rating
p <- function(x, y) mean(y == x)
rating <- seq(0.5,5,0.5)

# Estimate the probability of each rating with Monte Carlo simulation
B <- 10^3
simulation <- replicate(B, {
  s <- sample(train_set$rating, 100, replace = TRUE)
  sapply(rating, p, y= s)
})
prob <- sapply(1:nrow(simulation), function(x) mean(simulation[x,]))

# Predict random ratings
predicted_random <- sample(rating, size = nrow(test_set), 
                           replace = TRUE, prob = prob)

# Create a table with the error results
result <- tibble(Method = "Project Goal", RMSE = 0.8649, MSE = NA, MAE = NA)
result <- bind_rows(result, 
                    tibble(Method = "Random prediction", 
                           RMSE = RMSE(test_set$rating, predicted_random),
                           MSE  = MSE(test_set$rating, predicted_random),
                           MAE  = MAE(test_set$rating, predicted_random)))
```

```{r}
result
```

*I observe that the RMSE of random prediction is very high. So now ,I use Linear prediction Model.

**Initial Prediction**
  
  *The initial prediction is just the mean of the ratings, μ and some error (ŷ =μ+ϵi).*
  
  ```{r}
# Mean of observed values
mu <- mean(train_set$rating)

# Update the error table  
result <- bind_rows(result, 
                    tibble(Method = "Mean", 
                           RMSE = RMSE(test_set$rating, mu),
                           MSE  = MSE(test_set$rating, mu),
                           MAE  = MAE(test_set$rating, mu)))
```

```{r}
result
```

*As I know that all movies do not get ratings as well as all users do not do rating for all movies so there is some variation from movie to movie and user to user, I added movie variation bi and user variattion bu in linear model*
  
  ```{r}
# Movie effects (bi)
bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
head(bi)
```

**User effect**
  
  ```{r}
# User effect (bu)
bu <- train_set %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
```

**Movie and User both effect on Prediction**
  
  ```{r}
prediction_bi_bu <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
```

**Let see what is value of RMSE after miovie and user effect.**
  
  ```{r}
result <- bind_rows(result, 
                    tibble(Method = "Mean + bi + bu", 
                           RMSE = RMSE(test_set$rating, prediction_bi_bu),
                           MSE  = MSE(test_set$rating, prediction_bi_bu),
                           MAE  = MAE(test_set$rating, prediction_bi_bu)))
```

```{r}
result
```

*Results shows lots of improvement in RMSE value. We achieved the project goal value.*
  
  
  **Conclusion**
  
  *I started collecting and preparing the dataset for analysis, then I explored the information todevelop the model.*
  
  *Next, I started with probability model then a random model that predicts the rating based on the probability distribution of each rating. This model did not provide desiredresult.*
  
  *After that, I used  linear model the mean of the observed ratings. From there, I added movie and user effects. With  linear model ,I achieved the RMSE of 0.8648177, successfully passing the target of 0.8649.*
  
  
  **Limitations**
  
  *Some machine learning algorithms are computationally expensive to run in a regular laptop and therefore I was unable to test. The required amount of memory far exceeded the available in a regular laptop, even with increased virtual memory.*
  
  *Only two predictors are used, the movie and user information, not considering other features. Modern recommendation system models use many predictors, such as genres, bookmarks, playlists, etc.*
  
  *There is no initial recommendation for a new user or for users that usually don’t rate movies. Algorithms that uses several features as predictors can overcome this issue.*
  
  
  **Future Work**
  
  *This report uses only simple models such as Random prediction, Linear model predictionthat to predict ratings. There are two other widely adopted approaches not discussed here: content-based and collaborative filtering. The recommenderlab package implements these methods and provides an environment to build and test recommendation systems.*
  
  **References**
  
  *Rafael A. Irizarry (2019), Introduction to Data Science: Data Analysis and Prediction Algorithms with R*
  
  [This link](https://www.edx.org/professional-certificate/harvardx-data-science↩)

[This link](https://movielens.org/↩)

[This link](https://cran.r-project.org/web/packages/available_packages_by_name.html↩)

[This link](https://rpubs.com/)

update.packages(ask = FALSE, checkBuilt = TRUE)  # update R packages
tinytex::tlmgr_update()  # update LaTeX packages
