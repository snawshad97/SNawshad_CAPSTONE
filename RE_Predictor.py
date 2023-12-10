import time
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import tabulate
from utils.RealEstateDataset import RealEstateDataset
from utils.helper_functions import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW

hyperparams = {
    'learning_rate': [1e-2, 1e-6],
    'batch_size': [16, 32],
    'num_epochs': [1, 2],
    'model_name': ['bert', 'roberta', 'distilbert', 'gpt2', 'xlnet', 'electra']
}

# Load and preprocess the dataset
df = pd.read_csv('datasets/datafiniti_set.csv', sep=',')
df['address'] = replace_blanks(df['address'].fillna('Unknown').astype(str))
df['appliances'] = replace_blanks(df['appliances'].fillna('Unknown').astype(str))
df['architecturalStyles'] = replace_blanks(df['architecturalStyles'].fillna('Unknown').astype(str))
df['brokers'] = replace_blanks(df['brokers'].fillna('Unknown').astype(str))
df['categories'] = replace_blanks(df['categories'].fillna('Unknown').astype(str))
df['city'] = replace_blanks(df['city'].fillna('Unknown').astype(str))
df['country'] = replace_blanks(df['country'].fillna('Unknown').astype(str))
df['county'] = replace_blanks(df['county'].fillna('Unknown').astype(str))
df['currentOwnerType'] = replace_blanks(df['currentOwnerType'].fillna('Unknown').astype(str))
df['descriptions'] = replace_blanks(df['descriptions'].fillna('Unknown').astype(str))

df['combined_text'] = df['address'] + ' ' + df['appliances'] + ' ' +\
df['architecturalStyles'] + ' ' + df['brokers'] + ' ' + df['categories'] + ' ' +\
df['city'] + ' ' + df['country'] + ' ' + df['county'] + ' ' +\
df['currentOwnerType'] + ' ' + df['descriptions'] + ' '

target = df['mostRecentPriceAmount'].astype(float)
features = df['combined_text']

mean_price = target.mean()
std_price = target.std()
normalized_target = (target - mean_price) / std_price

normalized_target_tensor = torch.tensor(normalized_target.values, dtype=torch.float32)
mean_value = torch.nanmean(normalized_target_tensor)
normalized_target_tensor = torch.where(torch.isfinite(normalized_target_tensor), normalized_target_tensor, mean_value)
nan_or_inf_indices = torch.where(~torch.isfinite(normalized_target_tensor))[0]
print("Indices with NaN or Inf:", nan_or_inf_indices)
target = normalized_target_tensor

assert torch.isfinite(normalized_target_tensor).all(), "Targets still contain NaN or Inf values"

# Split the dataset into training, validation, and test sets
train_features, test_features, train_target_np, test_target_np = train_test_split(
    features, normalized_target_tensor.numpy(), test_size=0.2, random_state=42)

train_features, val_features, train_target_np, val_target_np = train_test_split(
    train_features, train_target_np, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Reset indices for features
train_features = train_features.reset_index(drop=True)
val_features = val_features.reset_index(drop=True)
test_features = test_features.reset_index(drop=True)

# Convert numpy arrays back to tensors
train_target = torch.tensor(train_target_np, dtype=torch.float32)
val_target = torch.tensor(val_target_np, dtype=torch.float32)
test_target = torch.tensor(test_target_np, dtype=torch.float32)

# Set Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Initialize a dictionary to store metrics
all_metrics = {}

# Training and evaluation
results = []
for model_name in hyperparams['model_name']:
    tokenizer, model = create_model(model_name)
    model.to(device)

    # Dictionary to store metrics for the current model
    all_metrics[model_name] = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [], 'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []}

    for lr in hyperparams['learning_rate']:
        batch_sizes = [1] if model_name == 'gpt2' else hyperparams['batch_size']
        for batch_size in batch_sizes:
            for num_epochs in hyperparams['num_epochs']:
                start_time = time.time()
                print(f"Training {model_name} with LR={lr}, Batch Size={batch_size}, Epochs={num_epochs}")
                 # Dataloader for each set
                train_dataset = RealEstateDataset(train_features, train_target, tokenizer, max_length=512)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataset = RealEstateDataset(val_features, val_target, tokenizer, max_length=512)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                test_dataset = RealEstateDataset(test_features, test_target, tokenizer, max_length=512)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)

                # optimizer, scheduler, and training loop
                optimizer = AdamW(model.parameters(), lr=lr)
                scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
                train_loss, train_rmse, train_mae, train_r2 = train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device)
                val_loss, val_rmse, val_mae, val_r2 = evaluate_model(model, val_loader, device)
                test_loss, test_rmse, test_mae, test_r2 = evaluate_model(model, test_loader, device)

                results.append([model_name, lr, batch_size, num_epochs, train_loss, train_rmse, train_mae, train_r2, val_loss, val_rmse, val_mae, val_r2, test_loss, test_rmse, test_mae, test_r2])
                all_metrics[model_name]['train_loss'].append(train_loss)
                all_metrics[model_name]['val_loss'].append(val_loss)
                all_metrics[model_name]['train_rmse'].append(train_rmse)
                all_metrics[model_name]['val_rmse'].append(val_rmse)
                all_metrics[model_name]['train_mae'].append(train_mae)
                all_metrics[model_name]['val_mae'].append(val_mae)
                all_metrics[model_name]['train_r2'].append(train_r2)
                all_metrics[model_name]['val_r2'].append(val_r2)
                training_time = time.time() - start_time
                results.append([model_name, lr, batch_size, num_epochs, train_loss, train_rmse, train_mae, train_r2, val_loss, val_rmse, val_mae, val_r2, test_loss, test_rmse, test_mae, test_r2, time.time() - start_time])
    
    # Generate and save plots for the current model
    plot_metrics(all_metrics[model_name], model_name, 'results/plots')

# Convert results list to a DataFrame
results_df = pd.DataFrame(results, columns=["Model", "LR", "Batch Size", "Epochs", "Train Loss", "Train RMSE", "Train MAE", "Train R2", "Val Loss", "Val RMSE", "Val MAE", "Val R2", "Test Loss", "Test RMSE", "Test MAE", "Test R2", "Training Time"])

# Generate and save comparative plots
plot_comparison_bar(results_df, 'Test RMSE', 'results/plots')
plot_comparison_bar(results_df, 'Test MAE', 'results/plots')

# Assuming you have a DataFrame with training times and accuracy metrics for efficiency analysis
efficiency_df = results_df[['Model', 'Training Time', 'Test RMSE']]
plot_efficiency(efficiency_df, 'Training Time', 'Test RMSE', 'results/plots')

# Print tabulated results
print(tabulate.tabulate(results_df, headers='keys', tablefmt='psql'))

# Create the directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save the DataFrame to a CSV file
results_df.to_csv('results/Pre-Trained_Model_Results.csv', index=False)

# Create a dictionary with a couple of new test data points
data = {
    'address': ['123 Maple St', '456 Oak St', ''],
    'appliances': ['Dishwasher, Refrigerator', '', 'Washer, Dryer, Range'],
    'architecturalStyles': ['Victorian', 'Modern', ''],
    'brokers': ['', '', ''],
    'categories': ['House, Property', 'Condo, New Development', ''],
    'city': ['Springfield', 'Shelbyville', ''],
    'country': ['US', 'US', 'US'],
    'county': ['Greene', 'Shelby', ''],
    'currentOwnerType': ['INDIVIDUAL', 'CORPORATION', ''],
    'descriptions': [
        'A beautiful victorian home with a spacious garden.',
        'Modern condo with city views and a communal gym.',
        ''
    ]
}

# Convert the dictionary to a DataFrame
test_df = pd.DataFrame(data)

# Apply the preprocessing steps to the test dataframe
test_df['address'] = replace_blanks(test_df['address'].fillna('Unknown').astype(str))
test_df['appliances'] = replace_blanks(test_df['appliances'].fillna('Unknown').astype(str))
test_df['architecturalStyles'] = replace_blanks(test_df['architecturalStyles'].fillna('Unknown').astype(str))
test_df['brokers'] = replace_blanks(test_df['brokers'].fillna('Unknown').astype(str))
test_df['categories'] = replace_blanks(test_df['categories'].fillna('Unknown').astype(str))
test_df['city'] = replace_blanks(test_df['city'].fillna('Unknown').astype(str))
test_df['country'] = replace_blanks(test_df['country'].fillna('Unknown').astype(str))
test_df['county'] = replace_blanks(test_df['county'].fillna('Unknown').astype(str))
test_df['currentOwnerType'] = replace_blanks(test_df['currentOwnerType'].fillna('Unknown').astype(str))
test_df['descriptions'] = replace_blanks(test_df['descriptions'].fillna('Unknown').astype(str))

test_df['combined_text'] = test_df['address'] + ' ' + test_df['appliances'] + ' ' +\
test_df['architecturalStyles'] + ' ' + test_df['brokers'] + ' ' + test_df['categories'] + ' ' +\
test_df['city'] + ' ' + test_df['country'] + ' ' + test_df['county'] + ' ' +\
test_df['currentOwnerType'] + ' ' + test_df['descriptions'] + ' '

# Make sure to move the model to eval mode
model.eval()

# Prepare to store the predictions
test_df['predicted_price'] = 0

# Loop through each row in the dataframe to predict the price
for index, row in test_df.iterrows():
    # Encode the combined text data for the model
    encoded_input = tokenizer.encode_plus(
        row['combined_text'],
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    # Move tensors to the same device as model
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Predict the price using the model
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        predicted_price = output.logits.item() * std_price + mean_price
        
        # Store the predicted price in the dataframe
        test_df.at[index, 'predicted_price'] = predicted_price

# Print the dataframe with predicted prices
print(test_df[['address', 'predicted_price']])
