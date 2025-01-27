##### Data Preparation #####

# Load necessary libraries
library(caret)    # For model training and evaluation
library(factoextra)    # For PCA visualization
library(ROCR)     # For ROC and AUC
library(glmnet)   # For LASSO/Ridge regression
library(randomForest)  # For Random Forest
library(cluster)  # For silhouette score
library(plotly)   # For 3D plotting
library(car)      # For VIF calculation
library(xgboost)  # For XGBoost
library(pROC)  # For ROC curve visualization
library(reshape2)  # For data manipulation
library(FactoMineR)  # For PCA
library(cluster)     # For silhouette score
library(NbClust)     # For NbClust
library(clustMixType)    # For clustering mixed data types
library(kohonen)    # For SOM
library(fpc)    # For DBSCAN
library(dbscan)    # For DBSCAN
library(plotly)  # For 3D plotting
library(stats)   # For statistical functions
library(Rtsne)  # For t-SNE
library(umap)   # For UMAP
library(dplyr)  # For data manipulation

#set working directory
setwd("C:/Users/cibei/OneDrive/Desktop/Statistical and machine learning/SL_project/Unsupervized_supervized_analysis")

# Load the credit data
credit_data <- read.csv("credit_clean/clean_dataset.csv")

str(credit_data)

# Step 1: Convert categorical variables (Industry, Ethnicity, Citizen) into factors
credit_data$Industry <- factor(credit_data$Industry)
credit_data$Ethnicity <- factor(credit_data$Ethnicity)
credit_data$Citizen <- factor(credit_data$Citizen)
credit_data$Approved <- factor(credit_data$Approved)
credit_data$Gender <- factor(credit_data$Gender)
credit_data$Married <- factor(credit_data$Married)
credit_data$BankCustomer <- factor(credit_data$BankCustomer)
credit_data$PriorDefault <- factor(credit_data$PriorDefault)
credit_data$Employed <- factor(credit_data$Employed)
credit_data$DriversLicense <- factor(credit_data$DriversLicense)

# Check the structure of the dataset after converting categorical variables
str(credit_data)

#Identify numeric features for scaling, excluding the 'Approved' label
num_features <- sapply(credit_data, is.numeric)

#Scale numeric features (standardization: center and scale)
preProcValues <- preProcess(credit_data[, num_features], method = c("center", "scale"))

# Apply scaling to the dataset, excluding 'Approved'
credit_data_scaled <- credit_data
credit_data_scaled[, num_features] <- predict(preProcValues, credit_data[, num_features])

#One-hot encode categorical variables (Industry, Ethnicity, Citizen)
dummy_vars <- dummyVars(~ Industry + Ethnicity + Citizen, data = credit_data_scaled)

# Apply the one-hot encoding
one_hot_encoded_data <- predict(dummy_vars, newdata = credit_data_scaled)

# Combine one-hot encoded columns with the rest of the dataset, excluding original categorical columns
credit_data_final <- cbind(credit_data_scaled[, !(names(credit_data_scaled) %in% c("Industry", "Ethnicity", "Citizen"))], 
                           one_hot_encoded_data)
# Convert 'Approved' to a factor with more descriptive levels
credit_data_final$Approved <- factor(credit_data_final$Approved, levels = c(0, 1), labels = c("Rejected", "Approved"))


#Check the structure of the final dataset
str(credit_data_final)


### Check for multicollinearity ###

#Eliminate the BankCustomer and Married columns
credit_data_final <- credit_data_final[, !colnames(credit_data_final) %in% c("BankCustomer", "Married")]


# Create correlation matrix for numeric variables in credit_data_final
cor_matrix <- cor(credit_data_final[, sapply(credit_data_final, is.numeric)])

# Melt the correlation matrix into a format suitable for ggplot2
melted_cor <- melt(cor_matrix)

# Create a heatmap using ggplot2 and include text labels for the correlation values
ggplot(data = melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() + # Minimal theme for a clean look
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Matrix Heatmap", x = "", y = "")

# Find variables with high correlation coefficients
abs_cor_matrix <- abs(cor_matrix)
highly_correlated <- findCorrelation(abs_cor_matrix, cutoff = 0.7, names = TRUE)
print(highly_correlated)

# Eliminate highly correlated variables
credit_data_final <- credit_data_final[, !colnames(credit_data_final) %in% highly_correlated]


# Fit a logistic model to check VIFs using credit_data_final
initial_model <- glm(Approved ~ ., data = credit_data_final, family = binomial)

# Calculate VIFs for all variables in the model
vif(initial_model)

# check for multicollinearity
alias(initial_model)

# Remove aliased variables
credit_data_final <- credit_data_final[, !colnames(credit_data_final) %in% c("Industry.Utilities", "Ethnicity.White", "Citizen.Temporary")]

# Refit the logistic regression model without collinear variables
initial_model_reduced <- glm(Approved ~ ., data = credit_data_final, family = binomial)

# Calculate VIF values (using the reduced model from earlier)
vif_values <- vif(initial_model_reduced)

#Convert the VIF values into a data frame for plotting
vif_data <- data.frame(Variable = names(vif_values), VIF = vif_values)

# Plot the VIF values using ggplot2, with a threshold line at VIF = 5
ggplot(vif_data, aes(x = reorder(Variable, VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = 5, linetype = "dashed", color = "red", linewidth = 1) +  # Add a red dashed line at VIF = 5
  coord_flip() +  # Flip to make it horizontal
  labs(title = "VIF for Logistic Regression Model", x = "Variable", y = "VIF") +
  theme_minimal()


# Define the VIF threshold
vif_threshold <- 5

# Filter variables that have VIF greater than the threshold
high_vif_vars <- vif_data %>% filter(VIF > vif_threshold)

# Print or investigate variables with high VIF
print(high_vif_vars)

# Remove the high VIF variables from the credit_data_final dataset
credit_data_final <- credit_data_final[, !names(credit_data_final) %in% high_vif_vars]
str(credit_data_final)

# Save the preprocessed dataset to a CSV file
write.csv(credit_data_final, "credit_data_preprocessed.csv", row.names = FALSE)

# Read the preprocessed dataset from the CSV file
credit_data_final <- read.csv("credit_data_preprocessed.csv", header = TRUE)



colnames(credit_data_final)


#### Unsupservised Learning ####


#### PCA on Reduced Dataset ####

# Select the columns of interest
columns_of_interest <- c("Gender", "Age", "Debt", "YearsEmployed", "PriorDefault", 
                         "Employed", "CreditScore", "Income", "Approved", "Citizen.ByOtherMeans")

# Subset the dataset to include only these columns
credit_data_pca <- credit_data_final[, columns_of_interest]


#Prepare the features (exclude the 'Approved' column)
credit_features <- credit_data_pca[, !names(credit_data_pca) %in% "Approved"]

# Perform PCA on the standardized data
credit_pca <- prcomp(credit_features, center = TRUE, scale. = TRUE)
summary(credit_pca)

# Calculate the Eigenvalues and Cumulative Variance
eigenvalues <- (credit_pca$sdev)^2
cumulative_variance <- cumsum(eigenvalues) / sum(eigenvalues)

# Plot Cumulative Variance Explained
plot(cumulative_variance, type = "b", xlab = "Number of Principal Components", 
     ylab = "Cumulative Variance Explained", main = "Cumulative Variance Explained by PCA")
abline(h = 0.6, col = "red", lty = 2)  # Reference line at 60%

# Visualize Eigenvalues (Scree Plot)
fviz_eig(credit_pca, addlabels = TRUE, barfill = "#00AFBB", barcolor = "#FC4E07", ylim = c(0, 37))

# Select components with Eigenvalue > 1 (Kaiser Criterion)
eigenvalues
selected_components <- which(eigenvalues > 1)
print(paste("Selected components based on eigenvalue > 1: ", selected_components))

# Biplot of individuals and variables
fviz_pca_biplot(credit_pca, 
                geom.ind = "point", 
                col.ind = as.factor(credit_data_pca$Approved), 
                palette = c("#00AFBB", "#FC4E07"),  
                addEllipses = TRUE,  
                repel = TRUE,  
                col.var = "blue")  

# Contribution of variables to PC1, PC2 and PC3
fviz_contrib(credit_pca, choice = "var", axes = 1, top = 10)  
fviz_contrib(credit_pca, choice = "var", axes = 2, top = 10)  
fviz_contrib(credit_pca, choice = "var", axes = 3, top = 10)

# Display PCA loadings (how variables contribute to PCs)
pca_loadings <- credit_pca$rotation
pca_loadings

# Display PCA scores (transformed data)
pca_scores <- credit_pca$x
head(pca_scores)


#### Hierarchical Clustering with Dendrogram ####

str(credit_features)
h_credit <- credit_features
#Compute Gower's Distance for the mixed data 
binary_columns <- c(1, 5, 6, 9)  # Column indices of binary variables
gower_dist <- daisy(h_credit, metric = "gower", 
                    type = list(binary = binary_columns))

# Convert the Gower distance object to a matrix if necessary
gower_matrix <- as.matrix(gower_dist)

#Perform Hierarchical Clustering 
hclust_result <- hclust(as.dist(gower_matrix), method = "ward.D2")

#Plot the Dendrogram 
plot(hclust_result, labels = FALSE, main = "Dendrogram for Hierarchical Clustering", 
     xlab = "", sub = "", ylab = "Height")

#Elbow Point Identification 
# Plot the heights of the last 30 merges to find the elbow point
last_heights <- tail(hclust_result$height, 30)
num_clusters <- 31:2  # Number of clusters to evaluate

# Plot the last 30 values against the range 2-31
plot(num_clusters, last_heights, type = 'o', col = 'blue', pch = 16,
     main = 'Cluster Distance vs Number of Clusters',
     xlab = 'Number of Clusters', ylab = 'Cluster Distance')

# Add a red dot at the chosen elbow point (index may need adjustment)
elbow_index <- 23
points(num_clusters[elbow_index], last_heights[elbow_index], col = 'red', pch = 16)


# Use factoextra to visualize with clusters
k <- 4  # Specify the number of clusters
fviz_dend(hclust_result, k = k, cex = 0.5, color_labels_by_k = TRUE, rect = TRUE)

#Cut the Dendrogram to Form Clusters 
cluster_assignments <- cutree(hclust_result, k = k)
h_credit$cluster <- as.factor(cluster_assignments)  # Add the cluster assignments to the data

# Compute silhouette scores
silhouette_scores <- silhouette(cluster_assignments, dist(gower_matrix))

# Visualize silhouette plot
fviz_silhouette(silhouette_scores)

# Print average silhouette width
avg_silhouette_width <- mean(silhouette_scores[, 3])  # Extract silhouette widths
cat("Average silhouette width:", avg_silhouette_width, "\n")

# Separate numeric and categorical columns
numeric_vars <- h_credit %>% select(where(is.numeric), cluster)
categorical_vars <- h_credit %>% select(where(is.factor), where(is.character), cluster)

# Calculate summary statistics for numeric variables only
cluster_summary_numeric <- numeric_vars %>%
  group_by(cluster) %>%
  summarise(across(everything(), list(mean = ~mean(.), sd = ~sd(.)), .names = "{col}_{fn}"))

# Calculate mode for categorical variables
cluster_summary_categorical <- categorical_vars %>%
  group_by(cluster) %>%
  summarise(across(everything(), ~names(sort(table(.), decreasing = TRUE))[1], .names = "{col}_mode"))

# Print the full data frame with all columns
print(cluster_summary_numeric, width = Inf)
print(cluster_summary_categorical, width = Inf)



#### t-sne Dimensionality Reduction ####

# Select the features (excluding 'Approved') and the label
features <- credit_data_pca[, -which(colnames(credit_data_pca) == "Approved")]
label <- credit_data_pca$Approved

# Apply t-SNE on the selected features
set.seed(0)
tsne_result <- Rtsne(as.matrix(features), dims = 2, perplexity = 30, verbose = TRUE)

# Convert the t-SNE results to a data frame
tsne_data <- data.frame(tsne_result$Y)
colnames(tsne_data) <- c("X1", "X2")

# Combine t-SNE results with 'Approved' labels
tsne_combined <- cbind(tsne_data, Approved = label)

#Plot the t-SNE results with the Approved labels using Plotly 
fig <- plot_ly(data = tsne_combined, x = ~X1, y = ~X2, type = 'scatter', mode = 'markers', 
               split = ~Approved, colors = c('#636EFA','#EF553B'))

fig <- fig %>%
  layout(
    plot_bgcolor = "#e5ecf6",
    title = "t-SNE Visualization of Credit Data PCA",
    xaxis = list(title = "t-SNE 1"),
    yaxis = list(title = "t-SNE 2")
  )

# Display the plot
fig

# Apply K-Means Clustering on the t-SNE results
set.seed(42)
kmeans_result <- kmeans(tsne_data, centers = 3)  # Adjust 'centers' based on expected clusters

# Add cluster labels to the t-SNE data
tsne_combined$Cluster <- as.factor(kmeans_result$cluster)

# Plot the clusters using plotly
fig_cluster <- plot_ly(data = tsne_combined, x = ~X1, y = ~X2, color = ~Cluster, type = 'scatter', mode = 'markers')

fig_cluster <- fig_cluster %>%
  layout(
    plot_bgcolor = "#e5ecf6",
    title = "t-SNE Clustering Visualization",
    xaxis = list(title = "t-SNE 1"),
    yaxis = list(title = "t-SNE 2")
  )

# Display the clustered t-SNE plot
fig_cluster

#Silhouette Analysis 
# Calculate silhouette scores for k-means clustering
silhouette_scores <- silhouette(kmeans_result$cluster, dist(tsne_data))

# Display silhouette summary
silhouette_summary <- summary(silhouette_scores)
print(silhouette_summary)

# Visualize silhouette plot
fviz_silhouette(silhouette_scores, label = TRUE, print.summary = TRUE) +
  ggtitle("Silhouette Plot for K-Means Clustering on t-SNE Data") +
  theme_minimal() +
  scale_fill_manual(values = c("#00AFBB", "#FC4E07", "#E7B800", "#2E9FDF"))

#Analyze the Distribution of Labels Across Clusters
# Analyze distribution of Approved status within each t-SNE cluster
cluster_distribution <- table(tsne_combined$Approved, tsne_combined$Cluster)
print(cluster_distribution)

#Density Plot 
# Visualize density of t-SNE clusters for each label using ggplot2
ggplot(tsne_combined, aes(x = X1, y = X2, color = credit_data_famd$Approved)) +
  geom_density2d() +
  geom_point(aes(shape = credit_data_famd$Approved)) +
  ggtitle("Density Plot of t-SNE Clusters") +
  theme_minimal()


#### UMAP Dimensionality Reduction ####

# Exclude the 'Approved' column (assuming other columns are already numeric and one-hot encoded)
credit_data_all_numeric <- credit_data_pca[, !colnames(credit_data_pca) %in% c("Approved")]

# Set UMAP configuration for 2D and 3D
umap_config <- umap.defaults
umap_config$n_neighbors <- 30  # Number of neighbors (you can adjust this)
umap_config$min_dist <- 0.05    # Minimum distance between points

# 1. Apply UMAP for 2D projection
set.seed(42)  # Ensure reproducibility
umap_result_2d <- umap(credit_data_pca, config = umap_config)

# Extract UMAP 2D layout and combine with the Approved labels
umap_2d_layout <- data.frame(umap_result_2d$layout)
colnames(umap_2d_layout) <- c("UMAP1", "UMAP2")
umap_2d_layout$Approved <- credit_data_famd$Approved


str(umap_2d_layout)

#Create a 2D UMAP plot
fig_umap_2d <- plot_ly(umap_2d_layout, x = ~UMAP1, y = ~UMAP2, 
                       color = ~Approved, 
                       colors = c('#FF7F7F', '#77B5FE'), 
                       type = 'scatter', mode = 'markers') %>%
  layout(
    plot_bgcolor = "#e5ecf6",
    legend = list(title = list(text = 'Approval Status')),
    xaxis = list(title = "UMAP Component 1"),
    yaxis = list(title = "UMAP Component 2")
  )


# Show 2D UMAP plot
fig_umap_2d

#Apply UMAP for 3D projection
umap_result_3d <- umap(credit_data_all_numeric, n_components = 3, config = umap_config)

# Extract UMAP 3D layout and combine with the Approved labels
umap_3d_layout <- data.frame(umap_result_3d$layout)
colnames(umap_3d_layout) <- c("UMAP1", "UMAP2", "UMAP3")
umap_3d_layout$Approved <- credit_data_famd$Approved

# Create a 3D UMAP plot
fig_umap_3d <- plot_ly(umap_3d_layout, x = ~UMAP1, y = ~UMAP2, z = ~UMAP3, 
                       color = ~Approved, 
                       colors = c('#FF7F7F', '#77B5FE'), 
                       type = 'scatter3d', mode = 'markers') %>%
  layout(
    scene = list(
      xaxis = list(title = "UMAP Component 1"),
      yaxis = list(title = "UMAP Component 2"),
      zaxis = list(title = "UMAP Component 3")
    )
  )

# Show 3D UMAP plot
fig_umap_3d



#### supervised learning ####

##### Set up k-Fold Cross-Validation #####

set.seed(123)
trainIndex <- createDataPartition(credit_data_final$Approved, p = 0.8, list = FALSE)
train_data <- credit_data_final[trainIndex, ]
test_data <- credit_data_final[-trainIndex, ]





#### Logistic Regression with glmnet (with Train-Test Split) ####

# Create a copy of the dataset for glmnet analysis
credit_data_glmnet <- credit_data_final

# Convert categorical variables into dummy variables (one-hot encoding)
x_data <- model.matrix(Approved ~ ., data = credit_data_glmnet)[, -1]  # Exclude intercept column
y_data <- ifelse(credit_data_glmnet$Approved == "Approved", 1, 0) # Convert factor Approved to numeric 0/1

# Split the dataset into training and testing sets (70% train, 30% test)
set.seed(123)
train_index <- createDataPartition(y_data, p = 0.8, list = FALSE)
x_train <- x_data[train_index, ]
x_test <- x_data[-train_index, ]
y_train <- y_data[train_index]
y_test <- y_data[-train_index]

# Fit glmnet model with L1/L2 regularization using cross-validation (Elastic Net with alpha = 0.5)
set.seed(123)
cv_glmnet_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, nfolds = 10)  # alpha=0.5 for Elastic Net

# Print the best lambda
best_lambda <- cv_glmnet_model$lambda.min
cat("Best lambda (regularization strength):", best_lambda, "\n")

# Get the coefficients of the model
coefficients_glmnet <- coef(cv_glmnet_model, s = best_lambda)
print(coefficients_glmnet)

# Convert coefficients to a data frame and remove zero-coefficient variables
coefficients_df <- as.data.frame(as.matrix(coefficients_glmnet))
coefficients_df$Variable <- rownames(coefficients_df)

# Identify the variables with non-zero coefficients
non_zero_vars <- coefficients_df[coefficients_df$s1 != 0, "Variable"]
cat("Non-zero coefficient variables:", non_zero_vars, "\n")

# Remove the intercept from non-zero variables
non_zero_vars <- non_zero_vars[non_zero_vars != "(Intercept)"]

# Filter the training and test datasets to include only non-zero coefficient variables
x_train <- model.matrix(Approved ~ ., data = credit_data_glmnet)[train_index, non_zero_vars]
x_test <- model.matrix(Approved ~ ., data = credit_data_glmnet)[-train_index, non_zero_vars]

# Refit the logistic regression model on the updated training dataset using glmnet
cv_model_logit_final <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)


##### Plotting MAE vs Log Lambda #####

# Predict on the training data for all lambdas
train_predictions_all <- predict(cv_model_logit_final, newx = x_train, s = cv_model_logit_final$lambda, type = "response")

# Assess performance across all lambdas
performance_all_lambdas <- assess.glmnet(train_predictions_all, newy = y_train, family = "binomial")

# Filter only positive lambda values
positive_lambda_values <- cv_model_logit_final$lambda[cv_model_logit_final$lambda > 0]

# Filter corresponding MAE values for positive lambda values
positive_mae_values <- performance_all_lambdas$mae[cv_model_logit_final$lambda > 0]

# Plot Mean Absolute Error (MAE) vs Log Lambda for positive lambdas only
plot(log(positive_lambda_values), 
     positive_mae_values, 
     xlab = "Log Lambda", ylab = "Mean Absolute Error (MAE)", 
     main = "MAE vs Log Lambda for glmnet",
     xlim = range(log(positive_lambda_values)),  # Adjust x-axis limits
     ylim = range(positive_mae_values))  # Adjust y-axis to fit MAE values

# Add a vertical line for the best lambda value (if it is positive)
if(cv_model_logit_final$lambda.min > 0) {
  abline(v = log(cv_model_logit_final$lambda.min), lty = 2, col = "red")
}

##### Model Assessment Using assess.glmnet #####

# Assess performance on the test data
performance_measures <- assess.glmnet(cv_model_logit_final, newx = x_test, newy = y_test)
cat("\nPerformance Measures (on Test Data):\n")
print(performance_measures)

##### Confusion Matrix Using confusion.glmnet #####

# Generate the confusion matrix for the model on test data
conf_matrix_glmnet <- confusion.glmnet(cv_model_logit_final, newx = x_test, newy = y_test)
cat("\nConfusion Matrix:\n")
print(conf_matrix_glmnet)

##### ROC Curve and AUC Calculation Using roc.glmnet #####

# Generate ROC curve and compute AUC
roc_glmnet <- roc.glmnet(cv_model_logit_final, newx = x_test, newy = y_test)

# Plot the ROC curve
plot(roc_glmnet, main = "ROC Curve for Regularized Logistic Regression")

# Calculate AUC using the predicted probabilities
predicted_prob <- predict(cv_model_logit_final, s = best_lambda, newx = x_test, type = "response")

# Ensure that the response is properly formatted for AUC calculation
pred <- prediction(predicted_prob, y_test)

# Calculate AUC
auc_glmnet <- performance(pred, measure = "auc")
cat("AUC for glmnet Model:", auc_glmnet@y.values[[1]], "\n")


##### Random Forest with k-Fold Cross-Validation #####

# Load the necessary libraries
library(randomForest)
library(caret)  # For k-fold cross-validation
library(ROCR)   # For ROC/AUC


# Set up trainControl for k-fold cross-validation
train_control <- trainControl(method = "cv", 
                              number = 10, 
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary,
                              search = "grid",
                              savePredictions = "final")

##### Train the Random Forest Model Using k-Fold CV #####

# First model without tuning mtry
set.seed(123)
rf_model_cv <- train(Approved ~ ., 
                     data = train_data, 
                     method = "rf", 
                     trControl = train_control, 
                     ntree = 500,
                     metric = "ROC")

# Print the results
print(rf_model_cv)

tuneGrid <- expand.grid(.mtry = c(1: 27))


#search the best maxnodes value
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = rf_model_cv$bestTune$mtry)
for (maxnodes in c(5: 15)) {
  set.seed(123)
  rf_maxnode <- train(Approved~.,
                      data = train_data,
                      method = "rf",
                      metric = "ROC",
                      tuneGrid = tuneGrid,
                      trControl = train_control,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

#search the best ntrees value
store_ntrees <- list()
tuneGrid <- expand.grid(.mtry = rf_model_cv$bestTune$mtry)
for (ntrees in c(100, 200, 300, 400, 500, 600, 800, 900, 1000)) {
  set.seed(123)
  rf_ntrees <- train(Approved~.,
                      data = train_data,
                      method = "rf",
                      metric = "ROC",
                      tuneGrid = tuneGrid,
                      trControl = train_control,
                      importance = TRUE,
                      nodesize = 14,
                      ntree = ntrees)
  current_iteration <- toString(ntrees)
  store_ntrees[[current_iteration]] <- rf_ntrees
}
results_ntrees <- resamples(store_ntrees)
summary(results_ntrees)

# Train the final Random Forest model using the best parameters
set.seed(123)
fit_rf <- train(Approved ~ ., 
                data = train_data,
                method = "rf",
                metric = "ROC",
                trControl = train_control,
                importance = TRUE,
                nodesize = 14,
                ntree = 1000,
                tuneGrid = tuneGrid,
                maxnodes = 14)
                    

##### Evaluate the model on new data #####

#Evaluate on the test set
test_predictions <- predict(fit_rf, newdata = test_data)


#Confusion matrix for the test set
conf_matrix <- confusionMatrix(test_predictions, test_data$Approved)
print(conf_matrix)

# Extract accuracy, precision, recall, and F1-score
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
specificity <- conf_matrix$byClass['Specificity']
f1_score <- 2 * ((precision * recall) / (precision + recall))

cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("Specificity: ", specificity, "\n")
cat("F1-Score: ", f1_score, "\n")


# Predict probabilities for the positive class
prob_predictions <- predict(fit_rf, newdata = test_data, type = "prob")[, "Approved"]

# Create a prediction object for ROCR
pred <- prediction(prob_predictions, test_data$Approved)

# Create a performance object for ROC curve
roc_performance <- performance(pred, measure = "tpr", x.measure = "fpr")

# Plot variable importance
var_imp <- varImp(fit_rf)
plot(var_imp, main = "Variable Importance (Random Forest)")

# View top variables
print(var_imp)


##### Xgboost with k-Fold Cross-Validation #####

# Prepare the dataset (convert to matrix and separate labels)
X_data <- model.matrix(Approved ~ ., data = credit_data_final)[, -1]  # Convert features to matrix (exclude intercept)
y_data <- ifelse(credit_data_glmnet$Approved == "Approved", 1, 0) # Convert factor Approved to numeric 0/1

# 80/20 split
set.seed(123)  # Ensure reproducibility
split_indices <- createDataPartition(y_data, p = 0.8, list = FALSE)  # 80% train, 20% test split
X_train <- X_data[split_indices, ]
X_test <- X_data[-split_indices, ]
y_train <- y_data[split_indices]
y_test <- y_data[-split_indices]

# Confirm the structure of the training and test sets
cat("Training Set Size:", dim(X_train), "\n")  # Check training data size
cat("Test Set Size:", dim(X_test), "\n")  # Check test data size

# Train a basic/default XGBoost model
default_model <- xgboost(data = as.matrix(X_train),  # XGBoost requires the data as a matrix
                         label = y_train,            # Training labels
                         booster = "gbtree",         # Use tree-based models
                         objective = "binary:logistic",  # For binary classification
                         nrounds = 100,              # Number of boosting rounds
                         verbose = 0)                # Silence output

# Predict on test set using default model
y_pred <- predict(default_model, as.matrix(X_test), type = "response") > 0.5  # Threshold prediction at 0.5
accuracy <- sum(y_pred == y_test) / length(y_test)  # Calculate accuracy
print(paste("Accuracy (Default Model):", accuracy))

# Hyperparameter tuning using a grid search
hyperparam_grid <- expand.grid(
  nrounds = seq(from = 100, to = 300, by = 100),  # Number of boosting rounds
  eta = c(0.025, 0.05, 0.1, 0.3),  # Learning rates
  max_depth = c(4, 5, 6),  # Maximum tree depth
  gamma = c(0, 1, 2),  # Minimum loss reduction required to make a split
  colsample_bytree = c(0.5, 0.75, 1.0),  # Subsample ratio of columns
  min_child_weight = c(1, 3, 5),  # Minimum sum of instance weight (hessian) needed in a child
  subsample = 1  # Subsample ratio of the training instance
)

# Time-saving hyperparameter grid for XGBoost, with a narrower range
hyperparam_grid <- expand.grid(
  nrounds = seq(from = 250, to = 350, by = 50),  # Narrow range around 300
  eta = c(0.2, 0.3, 0.4),  # Focus around eta = 0.3
  max_depth = c(4, 5, 6),  # Test depths around 5
  gamma = c(1, 2, 3),  # Focus on gamma = 2, with some variability
  colsample_bytree = c(0.4, 0.5, 0.6),  # Around 0.5
  min_child_weight = c(1, 2),  # Focus around 1
  subsample = c(0.9, 1)  # Focus around 1
)

# Set up cross-validation control
tune_control <- caret::trainControl(
  method = "cv",  # Cross-validation method
  number = 10,     # 4-fold cross-validation
  verboseIter = FALSE,  # Silence training logs
  allowParallel = FALSE  # Disable parallel computing
)

# Perform hyperparameter tuning with XGBoost using the caret package
bst <- caret::train(
  x = X_train,
  y = as.factor(y_train),  # Convert target to factor for caret compatibility
  trControl = tune_control,  # Cross-validation control
  tuneGrid = hyperparam_grid,  # Hyperparameter grid
  method = "xgbTree",  # XGBoost with tree booster
  verbose = FALSE,
  verbosity = 0  # Silence output
)

# Print the best tuned hyperparameters
bst$bestTune

# Train the final XGBoost model using the best hyperparameters
final_model <- xgboost(data = as.matrix(X_train),
                       label = y_train,
                       booster = "gbtree",
                       objective = "binary:logistic",
                       nrounds = bst$bestTune$nrounds,  # Use the best nrounds
                       max_depth = bst$bestTune$max_depth,  # Use the best max_depth
                       colsample_bytree = bst$bestTune$colsample_bytree,
                       min_child_weight = bst$bestTune$min_child_weight,
                       subsample = bst$bestTune$subsample,
                       eta = bst$bestTune$eta,
                       gamma = bst$bestTune$gamma,
                       scale_pos_weight = 0.5,  # Adjust for class imbalance
                       verbose = 0)  # Silence output

# Predict on the test set
y_pred_prob <- predict(final_model, as.matrix(X_test))  # Predicted probabilities
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)  # Convert probabilities to binary predictions (0/1)

# Convert actual test labels (y_test) to factor for confusion matrix
y_test_factor <- as.factor(y_test)

# Create a confusion matrix using the `caret` package
conf_matrix_xgb <- confusionMatrix(as.factor(y_pred), y_test_factor)

# Print the confusion matrix
print(conf_matrix_xgb)

# Additional metrics
accuracy <- conf_matrix_xgb$overall['Accuracy']
precision <- conf_matrix_xgb$byClass['Pos Pred Value']
recall <- conf_matrix_xgb$byClass['Sensitivity']
specificity <- conf_matrix_xgb$byClass['Specificity']
f1_score <- 2 * ((precision * recall) / (precision + recall))

cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("Specificity: ", specificity, "\n")
cat("F1-Score: ", f1_score, "\n")


# Feature importance plot
importance_matrix <- xgb.importance(colnames(X_train), model = final_model)  # Get feature importance from the model
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")  # Plot importance


#### Neural Network with k-Fold Cross-Validation #####

# Prepare the dataset (convert to matrix and separate labels)
X_data <- model.matrix(Approved ~ ., data = credit_data_final)[, -1]  # Convert features to matrix (exclude intercept)
# Convert the Approved factor into binary numeric (0 for not_Approved, 1 for Approved)
y_data <- ifelse(credit_data_final$Approved == "Approved", 1, 0)

# 80/20 split
set.seed(123)  # Ensure reproducibility
split_indices <- createDataPartition(y_data, p = 0.8, list = FALSE)  # 80% train, 20% test split
X_train <- X_data[split_indices, ]
X_test <- X_data[-split_indices, ]
y_train <- y_data[split_indices]
y_test <- y_data[-split_indices]


# Define 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the neural network with 10-fold cross-validation
nn_cv_model <- train(X_train, as.factor(y_train),
                     method = "nnet",             # Neural network model
                     trControl = train_control,   # Cross-validation settings
                     preProcess = c("center", "scale"),  # Preprocess: center and scale the data
                     tuneLength = 5,              # Number of tuning parameter combinations to try
                     linout = FALSE)              # For classification

# Print the cross-validation results
print(nn_cv_model)


# Predict on the test set
predictions <- predict(nn_cv_model, newdata = X_test)

# Evaluate model performance using a confusion matrix
conf_matrix <- confusionMatrix(predictions, as.factor(y_test))

# Print the confusion matrix and accuracy
print(conf_matrix)

