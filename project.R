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


#set working directory
setwd("C:/Users/cibei/OneDrive/Desktop/Statistical and machine learning/SL_project")

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

# Step 2: Identify numeric features for scaling, excluding the 'Approved' label
num_features <- sapply(credit_data, is.numeric)

# Step 3: Scale numeric features (standardization: center and scale)
preProcValues <- preProcess(credit_data[, num_features], method = c("center", "scale"))

# Apply scaling to the dataset, excluding 'Approved'
credit_data_scaled <- credit_data
credit_data_scaled[, num_features] <- predict(preProcValues, credit_data[, num_features])

# Step 4: One-hot encode categorical variables (Industry, Ethnicity, Citizen)
dummy_vars <- dummyVars(~ Industry + Ethnicity + Citizen, data = credit_data_scaled)

# Apply the one-hot encoding
one_hot_encoded_data <- predict(dummy_vars, newdata = credit_data_scaled)

# Combine one-hot encoded columns with the rest of the dataset, excluding original categorical columns
credit_data_final <- cbind(credit_data_scaled[, !(names(credit_data_scaled) %in% c("Industry", "Ethnicity", "Citizen"))], 
                           one_hot_encoded_data)
# Convert 'Approved' to a factor with more descriptive levels
credit_data_final$Approved <- factor(credit_data_final$Approved, levels = c(0, 1), labels = c("Rejected", "Approved"))


# Step 5: Check the structure of the final dataset
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


# Define a VIF threshold (e.g., 5)
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


# Select the required columns from credit_data_final
selected_columns <- credit_data_final[, c("Age", "Debt", "YearsEmployed", 
                                          "CreditScore", "Income", "Approved")]

# Ensure 'Approved' is treated as a factor (categorical)
selected_columns$Approved <- as.factor(selected_columns$Approved)

# Define axis properties for the plot
axis = list(showline = FALSE, 
            zeroline = FALSE, 
            gridcolor = '#ffff', 
            ticklen = 4)

# Create the scatter plot matrix (splom) using plotly
fig <- selected_columns %>%  
  plot_ly()  %>%  
  add_trace(  
    type = 'splom',  # Scatterplot matrix type
    dimensions = list( 
      list(label = 'Age', values = ~Age),  
      list(label = 'Debt', values = ~Debt),  
      list(label = 'YearsEmployed', values = ~YearsEmployed),  
      list(label = 'CreditScore', values = ~CreditScore),  
      list(label = 'Income', values = ~Income)
    ),  
    color = ~Approved,  # Color by the 'Approved' variable
    colors = c('#636EFA','#EF553B')  # Define colors for 'Approved' and 'Rejected'
  )

# Customize the layout of the plot
fig <- fig %>% 
  layout( 
    legend = list(title = list(text = 'Approved Status')), 
    hovermode = 'closest', 
    dragmode = 'select', 
    plot_bgcolor = 'rgba(240,240,240,0.95)', 
    xaxis = axis, 
    yaxis = axis, 
    xaxis2 = axis, 
    xaxis3 = axis, 
    xaxis4 = axis, 
    yaxis2 = axis, 
    yaxis3 = axis, 
    yaxis4 = axis
  ) 

# Display the plot
fig


#### Unsupservised Learning ####


#### PCA on Reduced Dataset ####

# Select the columns of interest
columns_of_interest <- c("Gender", "Age", "Debt", "YearsEmployed", "PriorDefault", 
                         "Employed", "CreditScore", "Income", "Approved", "Citizen.ByOtherMeans")

# Subset the dataset to include only these columns
credit_data_pca <- credit_data_final[, columns_of_interest]


# Step 1: Prepare the features (exclude the 'Approved' column)
credit_features <- credit_data_pca[, !names(credit_data_pca) %in% "Approved"]

# Check the structure of the new dataset
str(credit_data_pca)

# Step 2: Standardize the data
credit_scaled <- scale(credit_features)

# Step 3: Perform PCA on the standardized data
credit_pca <- prcomp(credit_scaled, center = TRUE, scale. = TRUE)
summary(credit_pca)

# Step 4: Calculate the Eigenvalues and Cumulative Variance
eigenvalues <- (credit_pca$sdev)^2
cumulative_variance <- cumsum(eigenvalues) / sum(eigenvalues)

# Step 5: Plot Cumulative Variance Explained
plot(cumulative_variance, type = "b", xlab = "Number of Principal Components", 
     ylab = "Cumulative Variance Explained", main = "Cumulative Variance Explained by PCA")
abline(h = 0.6, col = "red", lty = 2)  # Reference line at 60%

# Step 6: Visualize Eigenvalues (Scree Plot)
fviz_eig(credit_pca, addlabels = TRUE, barfill = "#00AFBB", barcolor = "#FC4E07", ylim = c(0, 37))

# Step 7: Select components with Eigenvalue > 1 (Kaiser Criterion)
selected_components <- which(eigenvalues > 1)
print(paste("Selected components based on eigenvalue > 1: ", selected_components))

# Step 8: Biplot of individuals and variables
fviz_pca_biplot(credit_pca, 
                geom.ind = "point", 
                col.ind = as.factor(credit_data_reduced$Approved), 
                palette = c("#00AFBB", "#FC4E07"),  
                addEllipses = TRUE,  
                repel = TRUE,  
                col.var = "blue")  

# Step 9: Contribution of variables to PC1, PC2 and PC3
fviz_contrib(credit_pca, choice = "var", axes = 1, top = 10)  
fviz_contrib(credit_pca, choice = "var", axes = 2, top = 10)  
fviz_contrib(credit_pca, choice = "var", axes = 3, top = 10)

# Step 10: Display PCA loadings (how variables contribute to PCs)
pca_loadings <- credit_pca$rotation
head(pca_loadings)

# Step 11: Display PCA scores (transformed data)
pca_scores <- credit_pca$x
head(pca_scores)

#### apply FAMD ####

# Select the columns of interest
columns_of_interest <- c("Gender", "Age", "Debt", "YearsEmployed", "PriorDefault", 
                         "Employed", "CreditScore", "Income", "Approved", "Citizen.ByOtherMeans")

# Subset the dataset to include only these columns
credit_data_famd <- credit_data_final[, columns_of_interest]

# Transform categorical variables into factors
credit_data_famd$Gender <- factor(credit_data_famd$Gender, levels = c(0, 1), labels = c("Female", "Male"))
credit_data_famd$PriorDefault <- factor(credit_data_famd$PriorDefault, levels = c(0, 1), labels = c("No", "Yes"))
credit_data_famd$Employed <- factor(credit_data_famd$Employed, levels = c(0, 1), labels = c("No", "Yes"))
credit_data_famd$Approved <- factor(credit_data_famd$Approved, levels = c("Rejected", "Approved"))
credit_data_famd$Citizen.ByOtherMeans <- factor(credit_data_famd$Citizen.ByOtherMeans, levels = c(0, 1), labels = c("No", "Yes"))

# Check the structure of the dataset after the transformation
str(credit_data_famd)

# Exclude the target variable 'Approved' if you want to analyze only features
famd_data <- credit_data_famd[, !colnames(credit_data_famd) %in% c('Approved')]

#Eliminate the Industry_Aggregated and Ethnicity_Group columns
famd_data <- famd_data[, !colnames(famd_data) %in% c("Industry_Aggregated", "Ethnicity_Group")]

# Check the structure of the data to ensure factors were applied correctly
str(famd_data)

# Ensure the row names are unique for famd_data
rownames(famd_data) <- make.unique(as.character(1:nrow(famd_data)))

# Apply FAMD on the dataset
famd_result <- FAMD(famd_data, ncp = 11)  # ncp = number of dimensions to keep, adjust as necessary

# Summary of the FAMD results
summary(famd_result)

# Extract the individual coordinates (Dim 1 and Dim 2)
individuals_coords <- as.data.frame(famd_result$ind$coord[, 1:2])  # First two dimensions

# Add the 'Approved' column from your original dataset to the coordinates
individuals_coords$Approved <- credit_data_final$Approved

# Create a ggplot scatter plot for individuals based on FAMD dimensions
ggplot(individuals_coords, aes(x = Dim.1, y = Dim.2, color = Approved)) +
  geom_point(size = 2) +  # Add points
  theme_minimal() +  # Use a minimal theme for a clean look
  labs(title = "FAMD Individual Plot", x = "Dimension 1", y = "Dimension 2") +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +  # Customize color palette
  theme(legend.position = "right")  # Position the legend


# Plot the variables factor map
fviz_famd_var(famd_result, repel = TRUE)

# Eigenvalues: Percentage of variance explained by each dimension
fviz_screeplot(famd_result, 
               addlabels = TRUE,            # Add labels on each bar
               ylim = c(0, 30),             # Adjust y-axis limits (based on your dataset)
               barfill = "steelblue",       # Change the color of the bars
               barcolor = "black",          # Add a black outline to the bars
               ggtheme = theme_minimal(),   # Use a minimal theme for a cleaner look
               title = "Scree Plot of FAMD Dimensions", # Add a descriptive title
               xlab = "Dimensions",         # Set the x-axis label
               ylab = "Percentage of Explained Variance" # Set the y-axis label
) + theme(
  plot.title = element_text(hjust = 0.5, size = 14, face = "bold"), # Center and style the title
  axis.title.x = element_text(size = 12, face = "bold"),            # Customize x-axis label
  axis.title.y = element_text(size = 12, face = "bold"),            # Customize y-axis label
  axis.text.x = element_text(angle = 45, hjust = 1)                 # Rotate x-axis text for clarity
)

# Contributions of variables to the dimensions
fviz_contrib(famd_result, choice = "var", axes = 1, top = 10)
fviz_contrib(famd_result, choice = "var", axes = 2, top = 10)
fviz_contrib(famd_result, choice = "var", axes = 3, top = 10)

#### Hierarchical Clustering with Dendrogram ####

# 1. Compute Gower's Distance for the mixed data
gower_dist <- daisy(famd_data, metric = "gower")

# Convert the Gower distance object to a matrix if necessary
gower_matrix <- as.matrix(gower_dist)

# 2. Perform Hierarchical Clustering using the 'ward.D2' method (can use other methods like 'complete', 'average')
hclust_result <- hclust(as.dist(gower_matrix), method = "ward.D2")

# 3. Plot the Dendrogram to visualize the hierarchical clustering
plot(hclust_result, labels = FALSE, main = "Dendrogram for Hierarchical Clustering", xlab = "", sub = "", ylab = "Height")

# 4. Use factoextra to visualize cluster6s in a more refined way
k <- 7  # Specify the number of clusters
fviz_dend(hclust_result, k = k, cex = 0.5, color_labels_by_k = TRUE, rect = TRUE)



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

### 1. Plot the t-SNE results with the Approved labels using Plotly ###
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

### 2. Apply K-Means Clustering on the t-SNE results ###
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

### 3. Silhouette Analysis ###
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

### 4. Analyze the Distribution of Labels Across Clusters ###
# Analyze distribution of Approved status within each t-SNE cluster
cluster_distribution <- table(tsne_combined$Approved, tsne_combined$Cluster)
print(cluster_distribution)

### 5. Density Plot ###
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

# 2. Create a 2D UMAP plot
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

# 3. Apply UMAP for 3D projection
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

#### Logistic Regression with glm ####

#Define a function that performs K-Fold cross-validation for logistic regression
logistic_regression_cv <- function(data, formula, k = 10) {
  set.seed(123) # Set seed for reproducibility
  
  # Set up trainControl for k-fold cross-validation
  train_control <- trainControl(method = "cv", 
                                number = k, 
                                classProbs = TRUE, 
                                summaryFunction = twoClassSummary,
                                search = "grid",
                                savePredictions = "final")
  
  # Train the logistic regression model using k-fold cross-validation
  logit_model_cv <- train(formula, 
                          data = data, 
                          method = "glm", 
                          trControl = train_control, 
                          metric = "ROC",
                          family = "binomial")
  
  return(logit_model_cv)
}

#Perform 10-fold cross-validation for logistic regression
cv_model_logit <- logistic_regression_cv(train_data, Approved ~ .)

#Print the cross-validated results
print(cv_model_logit)

#Extract and display the cross-validation results
cv_results <- cv_model_logit$results
print(cv_results)

#Extract the best model from the cross-validated results
best_model_logit <- cv_model_logit$finalModel
print(best_model_logit)

#Summarize the best model
summary(best_model_logit)




#### Logistic Regression with Significant Variables ####

# Define a new formula with only the significant variables
significant_formula <- Approved ~ PriorDefault + CreditScore + Income + 
  Industry.ConsumerDiscretionary + Industry.ConsumerStaples + Industry.ConsumerStaples +
  Industry.Real.Estate + Industry.Research +  Ethnicity.Latino + Ethnicity.Other

# Use the existing logistic_regression_cv function with the new formula
cv_model_logit_significant <- logistic_regression_cv(train_data, significant_formula)

# Print the cross-validated results
print(cv_model_logit_significant)

# Extract and display the cross-validation results
cv_results_significant <- cv_model_logit_significant$results
print(cv_results_significant)

# Extract the best model from the cross-validated results
best_model_logit_significant <- cv_model_logit_significant$finalModel
print(summary(best_model_logit_significant))

# Predict on test data
predicted_probabilities <- predict(best_model_logit_significant, newdata = test_data, type = "response")

# Ensure the labels match the actual factor levels in the data
predictions <- ifelse(predicted_probabilities > 0.5, "Approved", "Rejected")

# Convert predictions to a factor with the same levels as test_data$Approved
predictions <- factor(predictions, levels = levels(test_data$Approved))

# Convert test_data$Approved to a factor with the correct levels
test_data$Approved <- factor(test_data$Approved, levels = c("Approved", "Rejected"))
# Convert predictions to factor with the same levels as test_data$Approved
predictions <- factor(predictions, levels = c("Approved", "Rejected"))


# Evaluate model using confusion matrix
confusionMatrix(predictions, test_data$Approved)




#### Logistic Regression with glmnet (with Train-Test Split) ####

# Create a copy of the dataset for glmnet analysis
credit_data_glmnet <- credit_data_final

# Convert categorical variables into dummy variables (one-hot encoding)
x_data <- model.matrix(Approved ~ ., data = credit_data_glmnet)[, -1]  # Exclude intercept column
y_data <- as.numeric(credit_data_glmnet$Approved) - 1  # Convert factor Approved to numeric 0/1

# Split the dataset into training and testing sets (70% train, 30% test)
set.seed(123)
train_index <- createDataPartition(y_data, p = 0.7, list = FALSE)
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

# Plot the Mean Absolute Error (MAE) against Log Lambda
plot(cv_model_logit_final$lambda, performance_all_lambdas$mae, log = "x", xlab = "Log Lambda", ylab = "Mean Absolute Error (MAE)", 
     main = "MAE vs Log Lambda for glmnet")
abline(v = log(cv_model_logit_final$lambda.min), lty = 2, col = "red")

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
                maxnodes = 15)
                    

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

library(ROCR)

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
y_data <- as.numeric(credit_data_final$Approved) - 1  # Convert factor Approved to numeric 0/1 for binary classification

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

# Set up cross-validation control
tune_control <- caret::trainControl(
  method = "cv",  # Cross-validation method
  number = 4,     # 4-fold cross-validation
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

# Evaluate the final model on the test set
y_pred <- predict(final_model, as.matrix(X_test), type = "response") > 0.5  # Threshold predictions at 0.5

accuracy <- sum(y_pred == y_test) / length(y_test)  # Calculate accuracy
print(paste("Accuracy (Final Model):", accuracy))

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

