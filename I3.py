import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Set matplotlib font support
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. Load data
print("Loading data...")
df = pd.read_csv('Life Expectancy Data.csv')

# 2. Coerce data types
print("Converting data types...")
# Convert numeric columns to numeric types
numeric_columns = ['Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                   'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 
                   'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 
                   ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', 
                   ' thinness 5-9 years', 'Income composition of resources', 'Schooling']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ensure Year is integer type
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')

# 3. Compute basic statistics for numeric columns
print("Computing basic statistics...")
numeric_stats = df[numeric_columns].describe()
print("Basic statistics computed for numeric columns.")

# 4. Create image 1: Variable overview table
print("Creating variable overview table...")
fig, ax = plt.subplots(figsize=(16, 10))

# Prepare table data
table_data = []
for col in df.columns:
    dtype = str(df[col].dtype)
    missing_pct = (df[col].isnull().sum() / len(df)) * 100
    table_data.append([col, dtype, f"{missing_pct:.1f}%"])

# Create table
table = ax.table(cellText=table_data,
                colLabels=['Column Name', 'Data Type', 'Missing %'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Set table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Set title
ax.set_title('Variable Overview Table', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Save image at 1600x1000 resolution
plt.tight_layout()
plt.savefig('fig_2_2a_variable_overview.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

# 5. Create image 2: Life expectancy histogram
print("Creating life expectancy histogram...")
fig, ax = plt.subplots(figsize=(16, 10))

# Clean data, remove missing values
life_exp_data = df['Life expectancy '].dropna()

# Create histogram
ax.hist(life_exp_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

# Set title and labels
ax.set_title('Life Expectancy Distribution Histogram', fontsize=16, fontweight='bold')
ax.set_xlabel('Life Expectancy (years)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# Add statistical information
mean_val = life_exp_data.mean()
median_val = life_exp_data.median()
ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
ax.legend()

# Add grid
ax.grid(True, alpha=0.3)

# Save image at 1600x1000 resolution
plt.tight_layout()
plt.savefig('fig_2_2b_lifeexp_hist.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 6. Create image 3: Status bar chart
print("Creating status bar chart...")
fig, ax = plt.subplots(figsize=(16, 10))

# Calculate status counts
status_counts = df['Status'].value_counts()

# Create bar chart
bars = ax.bar(status_counts.index, status_counts.values, color=['lightcoral', 'lightblue'])

# Set title and labels
ax.set_title('Country Count by Status', fontsize=16, fontweight='bold')
ax.set_xlabel('Status', fontsize=12)
ax.set_ylabel('Number of Countries', fontsize=12)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontsize=12)

# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Save image at 1600x1000 resolution
plt.tight_layout()
plt.savefig('fig_2_2c_status_bar.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("All images have been successfully created and saved!")
print("Generated files:")
print("- fig_2_2a_variable_overview.png")
print("- fig_2_2b_lifeexp_hist.png") 
print("- fig_2_2c_status_bar.png")

# 7. Create image 4: Correlation heatmap
print("Creating correlation heatmap...")
fig, ax = plt.subplots(figsize=(12, 7))

# Calculate correlation matrix for numeric columns
corr_matrix = df[numeric_columns].corr()

# Create heatmap
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(range(len(numeric_columns)))
ax.set_yticks(range(len(numeric_columns)))
ax.set_xticklabels([col.strip() for col in numeric_columns], rotation=45, ha='right')
ax.set_yticklabels([col.strip() for col in numeric_columns])

# Add correlation values as text annotations
for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20)

# Set title
ax.set_title('Pearson Correlation Heatmap of Numeric Variables', fontsize=14, fontweight='bold')

# Save image
plt.tight_layout()
plt.savefig('fig_2_3a_corr_heatmap.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 8. Create image 5: GDP vs Life expectancy scatter plot
print("Creating GDP vs Life expectancy scatter plot...")
fig, ax = plt.subplots(figsize=(12, 7))

# Clean data - remove missing values
gdp_life_data = df[['GDP', 'Life expectancy ']].dropna()
x_data = gdp_life_data['GDP']
y_data = gdp_life_data['Life expectancy ']

# Create scatter plot
ax.scatter(x_data, y_data, alpha=0.6, s=20, color='blue')

# Add polynomial fit line (3rd order)
if len(x_data) > 3:  # Need at least 4 points for 3rd order polynomial
    # Sort data for smooth line
    sorted_indices = np.argsort(x_data)
    x_sorted = x_data.iloc[sorted_indices]
    y_sorted = y_data.iloc[sorted_indices]
    
    # Fit 3rd order polynomial
    poly_coeffs = np.polyfit(x_sorted, y_sorted, 3)
    poly_func = np.poly1d(poly_coeffs)
    
    # Create smooth line
    x_line = np.linspace(x_sorted.min(), x_sorted.max(), 100)
    y_line = poly_func(x_line)
    
    ax.plot(x_line, y_line, color='red', linewidth=2, label='3rd Order Polynomial Fit')

# Set labels and title
ax.set_xlabel('GDP (USD)', fontsize=12)
ax.set_ylabel('Life Expectancy (years)', fontsize=12)
ax.set_title('GDP vs Life Expectancy Scatter Plot', fontsize=14, fontweight='bold')

# Add legend
ax.legend()

# Add grid
ax.grid(True, alpha=0.3)

# Save image
plt.tight_layout()
plt.savefig('fig_2_3b_gdp_scatter.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 9. Create image 6: Adult Mortality vs Life expectancy scatter plot
print("Creating Adult Mortality vs Life expectancy scatter plot...")
fig, ax = plt.subplots(figsize=(12, 7))

# Clean data - remove missing values
mortality_life_data = df[['Adult Mortality', 'Life expectancy ']].dropna()
x_data = mortality_life_data['Adult Mortality']
y_data = mortality_life_data['Life expectancy ']

# Create scatter plot
ax.scatter(x_data, y_data, alpha=0.6, s=20, color='green')

# Set labels and title
ax.set_xlabel('Adult Mortality (per 1000 population)', fontsize=12)
ax.set_ylabel('Life Expectancy (years)', fontsize=12)
ax.set_title('Adult Mortality vs Life Expectancy Scatter Plot', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3)

# Save image
plt.tight_layout()
plt.savefig('fig_2_3c_adultmort_scatter.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Additional images have been successfully created and saved!")
print("New generated files:")
print("- fig_2_3a_corr_heatmap.png")
print("- fig_2_3b_gdp_scatter.png")
print("- fig_2_3c_adultmort_scatter.png")

# 10. Create image 7: Missing values bar chart
fig, ax = plt.subplots(figsize=(16, 10))

# Calculate missing counts for all columns
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]  # Only show columns with missing values

# Create bar chart
bars = ax.bar(range(len(missing_counts)), missing_counts.values, color='coral')

# Set labels and title
ax.set_xlabel('Columns', fontsize=12)
ax.set_ylabel('Missing Count', fontsize=12)
ax.set_title('Missing Values Count per Column', fontsize=16, fontweight='bold')

# Set x-axis labels
ax.set_xticks(range(len(missing_counts)))
ax.set_xticklabels([col.strip() for col in missing_counts.index], rotation=45, ha='right')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Save image at 1600x1000 resolution
plt.tight_layout()
plt.savefig('fig_2_4a_missing_bar.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 11. Create image 8: Multi-panel boxplots
fig, axes = plt.subplots(1, 3, figsize=(16, 10))

# Define columns for boxplots
boxplot_columns = ['Measles ', 'infant deaths', 'under-five deaths ']
column_labels = ['Measles', 'Infant Deaths', 'Under-five Deaths']

# Create boxplots for each column
for i, (col, label) in enumerate(zip(boxplot_columns, column_labels)):
    # Clean data - remove missing values
    clean_data = df[col].dropna()
    
    # Create boxplot
    axes[i].boxplot(clean_data, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
    
    # Set labels and title
    axes[i].set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Count', fontsize=10)
    axes[i].grid(True, alpha=0.3)

# Set overall title
fig.suptitle('Multi-panel Boxplots for Death-related Variables', fontsize=16, fontweight='bold')

# Save image at 1600x1000 resolution
plt.tight_layout()
plt.savefig('fig_2_4b_box_skewed.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 12. Create image 9: Year counts line plot
fig, ax = plt.subplots(figsize=(16, 10))

# Calculate record counts per year
year_counts = df['Year'].value_counts().sort_index()

# Filter for years 2000-2015
year_counts_filtered = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2015)]

# Create line plot
ax.plot(year_counts_filtered.index, year_counts_filtered.values, 
        marker='o', linewidth=2, markersize=6, color='darkgreen')

# Set labels and title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Record Counts per Year (2000-2015)', fontsize=16, fontweight='bold')

# Set x-axis ticks to show all years
ax.set_xticks(year_counts_filtered.index)
ax.set_xticklabels(year_counts_filtered.index, rotation=45)

# Add grid
ax.grid(True, alpha=0.3)

# Add value labels on points
for year, count in year_counts_filtered.items():
    ax.annotate(f'{count}', (year, count), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

# Save image at 1600x1000 resolution
plt.tight_layout()
plt.savefig('fig_2_4c_year_counts.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Final batch of images has been successfully created and saved!")
print("Latest generated files:")
print("- fig_2_4a_missing_bar.png")
print("- fig_2_4b_box_skewed.png")
print("- fig_2_4c_year_counts.png")

# 13. Column selection logic and audit
print("Creating column selection logic and audit...")

# Define column selection criteria
selected_cols = []
dropped_cols = []

# Analyze each column for selection
for col in df.columns:
    missing_pct = (df[col].isnull().sum() / len(df)) * 100
    
    # Selection criteria
    if col in ['Country', 'Year', 'Status']:
        # Keep essential categorical variables
        selected_cols.append(col)
        note = "Essential categorical variable"
    elif col == 'Life expectancy ':
        # Keep target variable
        selected_cols.append(col)
        note = "Target variable"
    elif col in numeric_columns:
        # For numeric columns, apply stricter criteria
        if missing_pct < 50:  # Keep if missing < 50%
            selected_cols.append(col)
            note = f"Missing {missing_pct:.1f}% - acceptable"
        else:
            dropped_cols.append(col)
            note = f"Missing {missing_pct:.1f}% - too high"
    else:
        # Drop other columns
        dropped_cols.append(col)
        note = "Non-essential variable"

# Create audit data
audit_data = []
for col in df.columns:
    missing_pct = (df[col].isnull().sum() / len(df)) * 100
    if col in selected_cols:
        decision = "Keep"
    else:
        decision = "Drop"
    
    # Determine note based on column type and missing percentage
    if col in ['Country', 'Year', 'Status']:
        note = "Essential categorical variable"
    elif col == 'Life expectancy ':
        note = "Target variable"
    elif col in numeric_columns:
        if missing_pct < 50:
            note = f"Missing {missing_pct:.1f}% - acceptable"
        else:
            note = f"Missing {missing_pct:.1f}% - too high"
    else:
        note = "Non-essential variable"
    
    audit_data.append([col, decision, f"{missing_pct:.1f}%", note])

# 14. Create audit table
fig, ax = plt.subplots(figsize=(16, 10))

# Create table
table = ax.table(cellText=audit_data,
                colLabels=['Column', 'Keep/Drop', 'Missing %', 'Note'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Set table style
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Color code the decision column
for i in range(1, len(audit_data) + 1):
    if audit_data[i-1][1] == "Keep":
        table[(i, 1)].set_facecolor('#90EE90')  # Light green
    else:
        table[(i, 1)].set_facecolor('#FFB6C1')  # Light pink

# Set title
ax.set_title('Column Selection Audit', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Save image
plt.tight_layout()
plt.savefig('fig_3_1a_audit.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 15. Create small multiples histogram grid
print("Creating small multiples histogram grid...")

# Select numeric columns from selected_cols
selected_numeric = [col for col in selected_cols if col in numeric_columns]

# Calculate grid dimensions
n_cols = len(selected_numeric)
n_rows = (n_cols + 2) // 3  # Arrange in 3 columns

fig, axes = plt.subplots(n_rows, 3, figsize=(16, 10))
if n_rows == 1:
    axes = axes.reshape(1, -1)

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Create histograms for each selected numeric column
for i, col in enumerate(selected_numeric):
    if i < len(axes_flat):
        # Clean data
        clean_data = df[col].dropna()
        
        # Create histogram
        axes_flat[i].hist(clean_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Set title and labels
        axes_flat[i].set_title(f'{col.strip()}', fontsize=10, fontweight='bold')
        axes_flat[i].set_xlabel('Value', fontsize=8)
        axes_flat[i].set_ylabel('Frequency', fontsize=8)
        
        # Add grid
        axes_flat[i].grid(True, alpha=0.3)

# Hide unused subplots
for i in range(len(selected_numeric), len(axes_flat)):
    axes_flat[i].set_visible(False)

# Set overall title
fig.suptitle('Small Multiples Histogram Grid for Selected Numeric Columns', 
             fontsize=16, fontweight='bold')

# Save image
plt.tight_layout()
plt.savefig('fig_3_1b_hists.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Column selection and audit completed!")
print("Selected columns:", len(selected_cols))
print("Dropped columns:", len(dropped_cols))
print("New generated files:")
print("- fig_3_1a_audit.png")
print("- fig_3_1b_hists.png")

# 16. Data cleaning and preprocessing
print("Starting data cleaning and preprocessing...")

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import copy

# Create a copy of the original dataframe for comparison
df_original = df.copy()

# Select only the chosen columns
df_clean = df[selected_cols].copy()

# Store missing values before imputation for comparison
missing_before = df_clean.isnull().sum()

# 17. Train/test split
print("Performing train/test split...")
# Use Year as stratification variable to ensure temporal distribution
X = df_clean.drop('Life expectancy ', axis=1)
y = df_clean['Life expectancy ']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_clean['Year']
)

# Combine features and target for imputation
train_data = X_train.copy()
train_data['Life expectancy '] = y_train

test_data = X_test.copy()
test_data['Life expectancy '] = y_test

# 18. Imputation
print("Performing imputation...")

# Identify numeric and categorical columns
numeric_cols_clean = [col for col in selected_cols if col in numeric_columns]
categorical_cols_clean = [col for col in selected_cols if col not in numeric_columns]

# Impute numeric columns with median
numeric_imputer = SimpleImputer(strategy='median')
train_data[numeric_cols_clean] = numeric_imputer.fit_transform(train_data[numeric_cols_clean])
test_data[numeric_cols_clean] = numeric_imputer.transform(test_data[numeric_cols_clean])

# Impute categorical columns with most frequent
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_cols_clean] = categorical_imputer.fit_transform(train_data[categorical_cols_clean].astype('object'))
test_data[categorical_cols_clean] = categorical_imputer.transform(test_data[categorical_cols_clean].astype('object'))

# Store missing values after imputation
missing_after = train_data.isnull().sum()

# 19. IQR-based outlier capping for count variables
print("Applying IQR-based outlier capping...")

# Define count variables (variables that represent counts)
count_vars = ['infant deaths', 'under-five deaths ', 'Measles ', 'Polio', 'Diphtheria ']

# Apply IQR capping to count variables
for col in count_vars:
    if col in train_data.columns:
        Q1 = train_data[col].quantile(0.25)
        Q3 = train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        train_data[col] = train_data[col].clip(lower=lower_bound, upper=upper_bound)
        test_data[col] = test_data[col].clip(lower=lower_bound, upper=upper_bound)

# 20. Create pre/post missingness comparison
print("Creating pre/post missingness comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Before imputation
missing_before_filtered = missing_before[missing_before > 0]
bars1 = ax1.bar(range(len(missing_before_filtered)), missing_before_filtered.values, color='coral')
ax1.set_title('Missing Values Before Imputation', fontsize=14, fontweight='bold')
ax1.set_xlabel('Columns', fontsize=12)
ax1.set_ylabel('Missing Count', fontsize=12)
ax1.set_xticks(range(len(missing_before_filtered)))
ax1.set_xticklabels([col.strip() for col in missing_before_filtered.index], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

# After imputation
missing_after_filtered = missing_after[missing_after > 0]
if len(missing_after_filtered) > 0:
    bars2 = ax2.bar(range(len(missing_after_filtered)), missing_after_filtered.values, color='lightgreen')
    ax2.set_xticks(range(len(missing_after_filtered)))
    ax2.set_xticklabels([col.strip() for col in missing_after_filtered.index], rotation=45, ha='right')
else:
    ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
             transform=ax2.transAxes, fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

ax2.set_title('Missing Values After Imputation', fontsize=14, fontweight='bold')
ax2.set_xlabel('Columns', fontsize=12)
ax2.set_ylabel('Missing Count', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels for after imputation
if len(missing_after_filtered) > 0:
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('fig_3_2a_missing_prepost.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 21. Create pre/post boxplot comparison
print("Creating pre/post boxplot comparison...")

# Select a few key numeric variables for boxplot comparison
boxplot_vars = ['Life expectancy ', 'Adult Mortality', 'GDP', 'Schooling']

fig, axes = plt.subplots(2, len(boxplot_vars), figsize=(16, 10))

for i, var in enumerate(boxplot_vars):
    if var in df_original.columns and var in train_data.columns:
        # Before cleaning (original data)
        original_data = df_original[var].dropna()
        axes[0, i].boxplot(original_data, patch_artist=True,
                          boxprops=dict(facecolor='lightcoral', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, i].set_title(f'{var.strip()} - Before', fontsize=10, fontweight='bold')
        axes[0, i].grid(True, alpha=0.3)
        
        # After cleaning (imputed data)
        cleaned_data = train_data[var]
        axes[1, i].boxplot(cleaned_data, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7),
                          medianprops=dict(color='darkgreen', linewidth=2))
        axes[1, i].set_title(f'{var.strip()} - After', fontsize=10, fontweight='bold')
        axes[1, i].grid(True, alpha=0.3)

# Set overall title
fig.suptitle('Pre/Post Cleaning Boxplot Comparison', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_3_2b_box_prepost.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 22. Save cleaned dataset
print("Saving cleaned dataset...")

# Combine train and test data for the final cleaned dataset
df_final_clean = pd.concat([train_data, test_data], ignore_index=True)

# Save to CSV
df_final_clean.to_csv('life_expectancy_clean.csv', index=False)

print("Data cleaning and preprocessing completed!")
print("Train set size:", len(train_data))
print("Test set size:", len(test_data))
print("Final cleaned dataset size:", len(df_final_clean))
print("New generated files:")
print("- fig_3_2a_missing_prepost.png")
print("- fig_3_2b_box_prepost.png")
print("- life_expectancy_clean.csv")

# 23. Feature Engineering
print("Creating new features...")

# Create new features for both train and test data
def create_new_features(data):
    """Create new derived features"""
    data_new = data.copy()
    
    # 1. Health expenditure ratio (percentage expenditure / GDP)
    if 'percentage expenditure' in data_new.columns and 'GDP' in data_new.columns:
        data_new['health_expenditure_ratio'] = data_new['percentage expenditure'] / (data_new['GDP'] + 1e-6)
    
    # 2. Child mortality rate (infant deaths + under-five deaths)
    if 'infant deaths' in data_new.columns and 'under-five deaths ' in data_new.columns:
        data_new['child_mortality_rate'] = data_new['infant deaths'] + data_new['under-five deaths ']
    
    # 3. Disease burden (sum of vaccine-preventable diseases)
    disease_cols = ['Measles ', 'Polio', 'Diphtheria ']
    available_disease_cols = [col for col in disease_cols if col in data_new.columns]
    if available_disease_cols:
        data_new['disease_burden'] = data_new[available_disease_cols].sum(axis=1)
    
    # 4. Economic development index (GDP * Income composition)
    if 'GDP' in data_new.columns and 'Income composition of resources' in data_new.columns:
        data_new['economic_dev_index'] = data_new['GDP'] * data_new['Income composition of resources']
    
    # 5. Health infrastructure score (BMI + Schooling + Income composition)
    health_cols = [' BMI ', 'Schooling', 'Income composition of resources']
    available_health_cols = [col for col in health_cols if col in data_new.columns]
    if available_health_cols:
        data_new['health_infrastructure_score'] = data_new[available_health_cols].mean(axis=1)
    
    # 6. Population density proxy (Population / GDP)
    if 'Population' in data_new.columns and 'GDP' in data_new.columns:
        data_new['population_density_proxy'] = data_new['Population'] / (data_new['GDP'] + 1e-6)
    
    # 7. Mortality ratio (Adult Mortality / Life expectancy)
    if 'Adult Mortality' in data_new.columns and 'Life expectancy ' in data_new.columns:
        data_new['mortality_ratio'] = data_new['Adult Mortality'] / (data_new['Life expectancy '] + 1e-6)
    
    # 8. Development gap (difference between developed and developing countries)
    if 'Status' in data_new.columns:
        data_new['is_developed'] = (data_new['Status'] == 'Developed').astype(int)
    
    # 9. Year trend (years since 2000)
    if 'Year' in data_new.columns:
        data_new['years_since_2000'] = data_new['Year'] - 2000
    
    # 10. Alcohol consumption per capita (Alcohol / Population)
    if 'Alcohol' in data_new.columns and 'Population' in data_new.columns:
        data_new['alcohol_per_capita'] = data_new['Alcohol'] / (data_new['Population'] + 1e-6)
    
    return data_new

# Apply feature engineering to both train and test data
train_data_feat = create_new_features(train_data)
test_data_feat = create_new_features(test_data)

# Combine for correlation analysis
df_feat = pd.concat([train_data_feat, test_data_feat], ignore_index=True)

# Identify new features (columns not in original selected_cols)
new_features = [col for col in df_feat.columns if col not in selected_cols]
print(f"Created {len(new_features)} new features:")
for feat in new_features:
    print(f"  - {feat}")

# 24. Recompute correlation with target
print("Computing correlations with target variable...")

# Calculate correlation with Life expectancy for all numeric features
numeric_features = df_feat.select_dtypes(include=[np.number]).columns.tolist()
correlations_with_target = df_feat[numeric_features].corr()['Life expectancy '].drop('Life expectancy ')

# Sort by absolute correlation
correlations_sorted = correlations_with_target.abs().sort_values(ascending=False)

print("Top correlations with Life expectancy:")
for feat, corr in correlations_sorted.head(10).items():
    original_corr = correlations_with_target[feat]
    print(f"  {feat}: {original_corr:.3f}")

# 25. Create new features histogram grid
print("Creating new features histogram grid...")

# Select new numeric features for histograms
new_numeric_features = [col for col in new_features if col in numeric_features]

if len(new_numeric_features) > 0:
    # Calculate grid dimensions
    n_features = len(new_numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Create histograms for each new feature
    for i, feat in enumerate(new_numeric_features):
        if i < len(axes_flat):
            # Clean data
            clean_data = df_feat[feat].dropna()
            
            # Create histogram
            axes_flat[i].hist(clean_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            
            # Set title and labels
            axes_flat[i].set_title(f'{feat}', fontsize=10, fontweight='bold')
            axes_flat[i].set_xlabel('Value', fontsize=8)
            axes_flat[i].set_ylabel('Frequency', fontsize=8)
            
            # Add grid
            axes_flat[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(new_numeric_features), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Set overall title
    fig.suptitle('New Features Histogram Grid', fontsize=16, fontweight='bold')
    
    # Save image
    plt.tight_layout()
    plt.savefig('fig_3_3a_newfeat_hists.png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
else:
    print("No new numeric features to plot")

# 26. Create new features correlation plot
print("Creating new features correlation plot...")

# Select top correlations for visualization
top_correlations = correlations_sorted.head(15)

fig, ax = plt.subplots(figsize=(16, 10))

# Create horizontal bar plot
y_pos = np.arange(len(top_correlations))
bars = ax.barh(y_pos, correlations_with_target[top_correlations.index], 
               color=['red' if x < 0 else 'green' for x in correlations_with_target[top_correlations.index]])

# Set labels and title
ax.set_yticks(y_pos)
ax.set_yticklabels([feat.strip() for feat in top_correlations.index])
ax.set_xlabel('Correlation with Life Expectancy', fontsize=12)
ax.set_title('Feature Correlations with Life Expectancy (Top 15)', fontsize=16, fontweight='bold')

# Add value labels
for i, (bar, corr) in enumerate(zip(bars, correlations_with_target[top_correlations.index])):
    width = bar.get_width()
    ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
            f'{corr:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)

# Add vertical line at zero
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add grid
ax.grid(True, alpha=0.3, axis='x')

# Invert y-axis to show highest correlations at top
ax.invert_yaxis()

# Save image
plt.tight_layout()
plt.savefig('fig_3_3b_newfeat_corr.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Feature engineering and correlation analysis completed!")
print("New generated files:")
print("- fig_3_3a_newfeat_hists.png")
print("- fig_3_3b_newfeat_corr.png")

# 27. Mock regions.csv and demonstrate left join
print("Creating mock regions.csv and performing left join...")

# Create mock regions data
regions_data = {
    'Country': [
        'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria',
        'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize',
        'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria',
        'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Central African Republic',
        'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba',
        'Cyprus', 'Czech Republic', 'Democratic Republic of Congo', 'Denmark', 'Djibouti', 'Dominican Republic',
        'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji',
        'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada',
        'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland',
        'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan',
        'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho',
        'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
        'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia',
        'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand',
        'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama',
        'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania',
        'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines',
        'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles',
        'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa',
        'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
        'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga',
        'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
        'United Arab Emirates', 'United Kingdom', 'United States of America', 'Uruguay', 'Uzbekistan',
        'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
    ],
    'Region': [
        'South Asia', 'Europe', 'North Africa', 'Sub-Saharan Africa', 'South America', 'Europe', 'Oceania', 'Europe',
        'Europe', 'Caribbean', 'Middle East', 'South Asia', 'Caribbean', 'Europe', 'Europe', 'Central America',
        'Sub-Saharan Africa', 'South Asia', 'South America', 'Europe', 'Sub-Saharan Africa', 'South America', 'Southeast Asia', 'Europe',
        'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Southeast Asia', 'Sub-Saharan Africa', 'North America', 'Sub-Saharan Africa', 'Sub-Saharan Africa',
        'Sub-Saharan Africa', 'South America', 'East Asia', 'South America', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Central America', 'Europe', 'Caribbean',
        'Europe', 'Europe', 'Sub-Saharan Africa', 'Europe', 'Sub-Saharan Africa', 'Caribbean',
        'South America', 'North Africa', 'Central America', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Europe', 'Sub-Saharan Africa', 'Oceania',
        'Europe', 'Europe', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Europe', 'Europe', 'Sub-Saharan Africa', 'Europe', 'Caribbean',
        'Central America', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'South America', 'Caribbean', 'Central America', 'Europe', 'Europe',
        'South Asia', 'Southeast Asia', 'Middle East', 'Middle East', 'Europe', 'Middle East', 'Europe', 'Caribbean', 'East Asia', 'Middle East',
        'Central Asia', 'Sub-Saharan Africa', 'Oceania', 'Middle East', 'Central Asia', 'Southeast Asia', 'Europe', 'Middle East', 'Sub-Saharan Africa',
        'Sub-Saharan Africa', 'North Africa', 'Europe', 'Europe', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'Southeast Asia', 'South Asia',
        'Sub-Saharan Africa', 'Europe', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'North America', 'Oceania', 'Europe', 'Europe', 'East Asia',
        'Europe', 'North Africa', 'Sub-Saharan Africa', 'Southeast Asia', 'Sub-Saharan Africa', 'South Asia', 'Europe', 'Oceania',
        'Central America', 'Sub-Saharan Africa', 'Sub-Saharan Africa', 'East Asia', 'Europe', 'Middle East', 'South Asia', 'Oceania', 'Central America',
        'Oceania', 'South America', 'South America', 'Southeast Asia', 'Europe', 'Europe', 'Middle East', 'Europe',
        'Europe', 'Sub-Saharan Africa', 'Caribbean', 'Caribbean', 'Caribbean',
        'Oceania', 'Europe', 'Sub-Saharan Africa', 'Middle East', 'Sub-Saharan Africa', 'Europe', 'Sub-Saharan Africa',
        'Sub-Saharan Africa', 'Southeast Asia', 'Europe', 'Europe', 'Oceania', 'Sub-Saharan Africa', 'Sub-Saharan Africa',
        'East Asia', 'Sub-Saharan Africa', 'Europe', 'South Asia', 'Sub-Saharan Africa', 'South America', 'Sub-Saharan Africa', 'Europe',
        'Europe', 'Middle East', 'Central Asia', 'Sub-Saharan Africa', 'Southeast Asia', 'Southeast Asia', 'Sub-Saharan Africa', 'Oceania',
        'Caribbean', 'North Africa', 'Europe', 'Central Asia', 'Oceania', 'Sub-Saharan Africa', 'Europe',
        'Middle East', 'Europe', 'North America', 'South America', 'Central Asia',
        'Oceania', 'South America', 'Southeast Asia', 'Middle East', 'Sub-Saharan Africa', 'Sub-Saharan Africa'
    ]
}

# Create regions DataFrame
regions_df = pd.DataFrame(regions_data)

# Save mock regions.csv
regions_df.to_csv('regions.csv', index=False)
print("Created mock regions.csv with", len(regions_df), "countries")

# 28. Perform left join
print("Performing left join...")

# Get unique countries from the main dataset
main_countries = set(df_feat['Country'].unique())
regions_countries = set(regions_df['Country'].unique())

# Perform left join
merged_df = df_feat.merge(regions_df, on='Country', how='left')

# 29. Analyze join results
print("Analyzing join results...")

# Find unmatched countries
unmatched_countries = main_countries - regions_countries
matched_countries = main_countries & regions_countries
extra_countries = regions_countries - main_countries

# Create join report data
join_report_data = [
    ['Total countries in main dataset', len(main_countries)],
    ['Total countries in regions dataset', len(regions_countries)],
    ['Successfully matched countries', len(matched_countries)],
    ['Unmatched countries in main dataset', len(unmatched_countries)],
    ['Extra countries in regions dataset', len(extra_countries)],
    ['Join success rate (%)', f"{len(matched_countries)/len(main_countries)*100:.1f}%"]
]

# Add unmatched countries details
if len(unmatched_countries) > 0:
    join_report_data.append(['', ''])  # Empty row
    join_report_data.append(['Unmatched countries:', ''])
    for country in sorted(list(unmatched_countries))[:10]:  # Show first 10
        join_report_data.append([country, 'Not found in regions'])
    if len(unmatched_countries) > 10:
        join_report_data.append([f'... and {len(unmatched_countries)-10} more', ''])

# Add extra countries details
if len(extra_countries) > 0:
    join_report_data.append(['', ''])  # Empty row
    join_report_data.append(['Extra countries in regions:', ''])
    for country in sorted(list(extra_countries))[:10]:  # Show first 10
        join_report_data.append([country, 'Not in main dataset'])
    if len(extra_countries) > 10:
        join_report_data.append([f'... and {len(extra_countries)-10} more', ''])

# 30. Create join report table
print("Creating join report table...")

fig, ax = plt.subplots(figsize=(16, 10))

# Create table
table = ax.table(cellText=join_report_data,
                colLabels=['Metric', 'Value'],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1])

# Set table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Color code different types of rows
for i in range(len(join_report_data)):
    if i < 6:  # Main statistics
        table[(i+1, 0)].set_facecolor('#E6F3FF')  # Light blue
        table[(i+1, 1)].set_facecolor('#E6F3FF')
    elif join_report_data[i][0] == 'Unmatched countries:' or join_report_data[i][0] == 'Extra countries in regions:':
        table[(i+1, 0)].set_facecolor('#FFE6E6')  # Light red
        table[(i+1, 1)].set_facecolor('#FFE6E6')
    elif join_report_data[i][0] == '':
        table[(i+1, 0)].set_facecolor('#FFFFFF')  # White
        table[(i+1, 1)].set_facecolor('#FFFFFF')

# Set title
ax.set_title('Left Join Report: Main Dataset vs Regions Dataset', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Save image
plt.tight_layout()
plt.savefig('fig_3_4a_join_report.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 31. Save merged dataset
print("Saving merged dataset...")

# Add Region column to the merged dataset
merged_df.to_csv('life_expectancy_merged.csv', index=False)

print("Left join demonstration completed!")
print("Join statistics:")
print(f"  - Main dataset countries: {len(main_countries)}")
print(f"  - Regions dataset countries: {len(regions_countries)}")
print(f"  - Successfully matched: {len(matched_countries)}")
print(f"  - Unmatched in main: {len(unmatched_countries)}")
print(f"  - Extra in regions: {len(extra_countries)}")
print(f"  - Success rate: {len(matched_countries)/len(main_countries)*100:.1f}%")
print("New generated files:")
print("- regions.csv")
print("- fig_3_4a_join_report.png")
print("- life_expectancy_merged.csv")

# 32. Build preprocessing pipeline with ColumnTransformer
print("Building preprocessing pipeline with ColumnTransformer...")

# Import additional sklearn modules
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Prepare data for preprocessing
# Use the merged dataset with features
X_full = merged_df.drop('Life expectancy ', axis=1)
y_full = merged_df['Life expectancy ']

# Identify column types
categorical_columns = ['Country', 'Status', 'Region']
numeric_columns_all = [col for col in X_full.columns if col not in categorical_columns]

print(f"Categorical columns: {len(categorical_columns)}")
print(f"Numeric columns: {len(numeric_columns_all)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns_all),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
    ],
    remainder='passthrough'
)

# Fit and transform the data
print("Fitting and transforming data...")
X_processed = preprocessor.fit_transform(X_full)

# Get feature names after transformation
numeric_feature_names = numeric_columns_all.copy()

# Get categorical feature names from OneHotEncoder
cat_encoder = preprocessor.named_transformers_['cat']
categorical_feature_names = []
for i, col in enumerate(categorical_columns):
    categories = cat_encoder.categories_[i]
    for cat in categories[1:]:  # Skip first category due to drop='first'
        categorical_feature_names.append(f"{col}_{cat}")

all_feature_names = numeric_feature_names + categorical_feature_names

print(f"Original features: {X_full.shape[1]}")
print(f"Processed features: {X_processed.shape[1]}")
print(f"Target shape: {y_full.shape}")

# 33. Save processed data
print("Saving processed data...")

# Save as numpy arrays
np.save('X.npy', X_processed)
np.save('y.npy', y_full.values)

# Save preprocessor for later use
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Processed data saved:")
print(f"  - X.npy: {X_processed.shape}")
print(f"  - y.npy: {y_full.shape}")
print(f"  - preprocessor.pkl: fitted transformer")

# 34. Create matrix schema preview
print("Creating matrix schema preview...")

# Create a small preview of the processed data
n_preview = min(10, X_processed.shape[0])
n_features_preview = min(15, X_processed.shape[1])

# Select subset for preview
X_preview = X_processed[:n_preview, :n_features_preview]
feature_names_preview = all_feature_names[:n_features_preview]

# Create preview data with row labels
preview_data = []
for i in range(n_preview):
    row = [f"Sample_{i+1}"] + [f"{val:.3f}" for val in X_preview[i]]
    preview_data.append(row)

# Add feature names as header
header = ["Sample"] + feature_names_preview

fig, ax = plt.subplots(figsize=(16, 10))

# Create table
table = ax.table(cellText=preview_data,
                colLabels=header,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Set table style
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)

# Color code different types of features
for i, feature_name in enumerate(feature_names_preview):
    col_idx = i + 1  # +1 because first column is Sample
    if any(cat_col in feature_name for cat_col in categorical_columns):
        # Categorical features (one-hot encoded)
        for row_idx in range(len(preview_data) + 1):
            table[(row_idx, col_idx)].set_facecolor('#FFE6E6')  # Light red
    else:
        # Numeric features (standardized)
        for row_idx in range(len(preview_data) + 1):
            table[(row_idx, col_idx)].set_facecolor('#E6F3FF')  # Light blue

# Set title
ax.set_title('Preprocessed Data Matrix Schema Preview', fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#E6F3FF', label='Numeric Features (Standardized)'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#FFE6E6', label='Categorical Features (One-Hot Encoded)')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

ax.axis('off')

# Save image
plt.tight_layout()
plt.savefig('fig_3_5a_matrix_schema.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 35. Summary statistics
print("Preprocessing pipeline completed!")
print("Summary:")
print(f"  - Original dataset shape: {merged_df.shape}")
print(f"  - Features after preprocessing: {X_processed.shape[1]}")
print(f"  - Numeric features: {len(numeric_feature_names)}")
print(f"  - Categorical features (one-hot): {len(categorical_feature_names)}")
print(f"  - Target samples: {len(y_full)}")
print("New generated files:")
print("- X.npy")
print("- y.npy")
print("- preprocessor.pkl")
print("- fig_3_5a_matrix_schema.png")

# 36. Train RandomForestRegressor and analyze feature importance
print("Training RandomForestRegressor...")

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split the processed data back into train/test
# We need to use the same indices as before
X_train_processed = X_processed[:len(train_data_feat)]
X_test_processed = X_processed[len(train_data_feat):]
y_train_processed = y_full[:len(train_data_feat)]
y_test_processed = y_full[len(train_data_feat):]

print(f"Training set shape: {X_train_processed.shape}")
print(f"Test set shape: {X_test_processed.shape}")

# Train RandomForestRegressor with default parameters
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model.fit(X_train_processed, y_train_processed)

# Make predictions
y_train_pred = rf_model.predict(X_train_processed)
y_test_pred = rf_model.predict(X_test_processed)

# Calculate metrics
train_mse = mean_squared_error(y_train_processed, y_train_pred)
test_mse = mean_squared_error(y_test_processed, y_test_pred)
train_r2 = r2_score(y_train_processed, y_train_pred)
test_r2 = r2_score(y_test_processed, y_test_pred)

print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# 37. Create feature importance plot
print("Creating feature importance plot...")

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create DataFrame for easier handling
importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# Select top 15 features
top_15_features = importance_df.head(15)

fig, ax = plt.subplots(figsize=(16, 10))

# Create horizontal bar plot
y_pos = np.arange(len(top_15_features))
bars = ax.barh(y_pos, top_15_features['importance'], color='skyblue', edgecolor='black')

# Set labels and title
ax.set_yticks(y_pos)
ax.set_yticklabels([feat.strip() for feat in top_15_features['feature']])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 15 Feature Importances from Random Forest', fontsize=16, fontweight='bold')

# Add value labels
for i, (bar, importance) in enumerate(zip(bars, top_15_features['importance'])):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
            f'{importance:.4f}', ha='left', va='center', fontsize=9)

# Add grid
ax.grid(True, alpha=0.3, axis='x')

# Invert y-axis to show highest importance at top
ax.invert_yaxis()

# Save image
plt.tight_layout()
plt.savefig('fig_4_1a_rf_importance.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 38. Create correlation-based feature dropping heatmap
print("Creating correlation-based feature dropping heatmap...")

# Calculate correlation matrix for numeric features only
numeric_features_original = [col for col in merged_df.columns if col not in ['Country', 'Status', 'Region', 'Life expectancy ']]
correlation_matrix = merged_df[numeric_features_original].corr()

# Find highly correlated pairs (threshold = 0.8)
high_corr_pairs = []
correlation_threshold = 0.8

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) >= correlation_threshold:
            high_corr_pairs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': corr_value
            })

# Create a matrix showing which features would be dropped
# For simplicity, we'll drop the second feature in each highly correlated pair
features_to_drop = set()
for pair in high_corr_pairs:
    features_to_drop.add(pair['feature2'])

# Create a binary matrix showing drop decisions
drop_matrix = pd.DataFrame(0, index=numeric_features_original, columns=numeric_features_original)
for pair in high_corr_pairs:
    drop_matrix.loc[pair['feature1'], pair['feature2']] = 1
    drop_matrix.loc[pair['feature2'], pair['feature1']] = 1

# Create heatmap
fig, ax = plt.subplots(figsize=(16, 10))

# Create heatmap
im = ax.imshow(drop_matrix.values, cmap='Reds', aspect='auto')

# Set ticks and labels
ax.set_xticks(range(len(numeric_features_original)))
ax.set_yticks(range(len(numeric_features_original)))
ax.set_xticklabels([col.strip() for col in numeric_features_original], rotation=45, ha='right')
ax.set_yticklabels([col.strip() for col in numeric_features_original])

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Would be Dropped (1) or Kept (0)', rotation=270, labelpad=20)

# Set title
ax.set_title(f'Correlation-based Feature Dropping Heatmap (threshold ≥ {correlation_threshold})', 
             fontsize=16, fontweight='bold')

# Add text annotations for high correlations
for pair in high_corr_pairs:
    idx1 = numeric_features_original.index(pair['feature1'])
    idx2 = numeric_features_original.index(pair['feature2'])
    ax.text(idx2, idx1, f'{pair["correlation"]:.2f}', 
            ha='center', va='center', color='white', fontsize=8, fontweight='bold')

# Save image
plt.tight_layout()
plt.savefig('fig_4_1b_corr_drop.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 39. Summary
print("Random Forest training and analysis completed!")
print("Model Performance:")
print(f"  - Training R²: {train_r2:.4f}")
print(f"  - Test R²: {test_r2:.4f}")
print(f"  - Training MSE: {train_mse:.4f}")
print(f"  - Test MSE: {test_mse:.4f}")
print("Feature Analysis:")
print(f"  - Total features: {len(all_feature_names)}")
print(f"  - Features with high correlation (≥{correlation_threshold}): {len(high_corr_pairs)} pairs")
print(f"  - Features that would be dropped: {len(features_to_drop)}")
print("New generated files:")
print("- fig_4_1a_rf_importance.png")
print("- fig_4_1b_corr_drop.png")

# 40. PCA Analysis on scaled features
print("Performing PCA analysis on scaled features...")

# Import PCA
from sklearn.decomposition import PCA

# Extract only numeric features for PCA (exclude one-hot encoded categorical features)
# We'll use the original numeric features before one-hot encoding
numeric_features_for_pca = [col for col in merged_df.columns if col not in ['Country', 'Status', 'Region', 'Life expectancy ']]
X_numeric = merged_df[numeric_features_for_pca].values

# Apply StandardScaler to numeric features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Fit PCA with 2 components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_numeric_scaled)

# Get Status labels for coloring
status_labels = merged_df['Status'].values

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# 41. Create PCA scatter plot colored by Status
print("Creating PCA scatter plot colored by Status...")

fig, ax = plt.subplots(figsize=(16, 10))

# Get unique status values and assign colors
unique_statuses = np.unique(status_labels)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
status_colors = {status: colors[i % len(colors)] for i, status in enumerate(unique_statuses)}

# Create scatter plot
for status in unique_statuses:
    mask = status_labels == status
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              c=status_colors[status], label=status, alpha=0.6, s=30)

# Set labels and title
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('PCA Scatter Plot: First Two Principal Components (Colored by Status)', 
             fontsize=16, fontweight='bold')

# Add legend
ax.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid
ax.grid(True, alpha=0.3)

# Add PCA component information
info_text = f'Total Explained Variance: {pca.explained_variance_ratio_.sum():.1%}\n'
info_text += f'PC1: {pca.explained_variance_ratio_[0]:.1%}\n'
info_text += f'PC2: {pca.explained_variance_ratio_[1]:.1%}'
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save image
plt.tight_layout()
plt.savefig('fig_4_2a_pca_scatter.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 42. Summary
print("PCA analysis completed!")
print("PCA Results:")
print(f"  - Original features: {X_numeric.shape[1]}")
print(f"  - PCA components: {X_pca.shape[1]}")
print(f"  - PC1 explained variance: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  - PC2 explained variance: {pca.explained_variance_ratio_[1]:.4f}")
print(f"  - Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print("New generated files:")
print("- fig_4_2a_pca_scatter.png")

# 43. Create diagnostic plots from training data only
print("Creating diagnostic plots from training data only...")

# Import additional sklearn modules for diagnostics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Use the merged dataset for diagnostics (training data only)
# We'll use the first part which corresponds to training data
train_size = len(train_data_feat)
X_train_diag = merged_df.iloc[:train_size].copy()
y_train_diag = merged_df.iloc[:train_size]['Life expectancy '].copy()

# Get top 3 predictors based on correlation with target
numeric_features_diag = [col for col in X_train_diag.columns if col not in ['Country', 'Status', 'Region', 'Life expectancy ']]
correlations_with_target_diag = X_train_diag[numeric_features_diag].corrwith(y_train_diag).abs().sort_values(ascending=False)
top_3_predictors = correlations_with_target_diag.head(3).index.tolist()

print(f"Top 3 predictors: {top_3_predictors}")

# 44. Create fig_6_1a_linearity_plot.png - Partial scatter + polynomial fit for top 3 predictors
print("Creating fig_6_1a_linearity_plot.png...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, predictor in enumerate(top_3_predictors):
    # Clean data - remove missing values
    clean_data = X_train_diag[[predictor, 'Life expectancy ']].dropna()
    x_data = clean_data[predictor]
    y_data = clean_data['Life expectancy ']
    
    # Create scatter plot
    axes[i].scatter(x_data, y_data, alpha=0.6, s=20, color='blue')
    
    # Add polynomial fit (2nd order)
    if len(x_data) > 2:
        # Sort data for smooth line
        sorted_indices = np.argsort(x_data)
        x_sorted = x_data.iloc[sorted_indices]
        y_sorted = y_data.iloc[sorted_indices]
        
        # Fit 2nd order polynomial
        poly_coeffs = np.polyfit(x_sorted, y_sorted, 2)
        poly_func = np.poly1d(poly_coeffs)
        
        # Create smooth line
        x_line = np.linspace(x_sorted.min(), x_sorted.max(), 100)
        y_line = poly_func(x_line)
        
        axes[i].plot(x_line, y_line, color='red', linewidth=2, label='2nd Order Polynomial Fit')
        
        # Calculate R²
        y_pred_poly = poly_func(x_sorted)
        r2_poly = r2_score(y_sorted, y_pred_poly)
        axes[i].text(0.05, 0.95, f'R² = {r2_poly:.3f}', transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    axes[i].set_xlabel(predictor.strip(), fontsize=10)
    axes[i].set_ylabel('Life Expectancy (years)', fontsize=10)
    axes[i].set_title(f'{predictor.strip()}\nvs Life Expectancy', fontsize=12, fontweight='bold')
    
    # Add legend
    axes[i].legend()
    
    # Add grid
    axes[i].grid(True, alpha=0.3)

# Set overall title
fig.suptitle('Partial Scatter Plots with Polynomial Fits for Top 3 Predictors', fontsize=16, fontweight='bold')

# Save image
plt.tight_layout()
plt.savefig('fig_6_1a_linearity_plot.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 45. Create fig_6_1b_outlier_influence.png - Residual vs fitted from baseline Linear Regression
print("Creating fig_6_1b_outlier_influence.png...")

# Prepare data for linear regression (numeric features only)
X_train_numeric = X_train_diag[numeric_features_diag].dropna()
y_train_numeric = y_train_diag.loc[X_train_numeric.index]

# Train baseline linear regression
lr_model = LinearRegression()
lr_model.fit(X_train_numeric, y_train_numeric)

# Make predictions
y_pred_lr = lr_model.predict(X_train_numeric)

# Calculate residuals
residuals = y_train_numeric - y_pred_lr

# Create residual vs fitted plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create scatter plot
ax.scatter(y_pred_lr, residuals, alpha=0.6, s=20, color='blue')

# Add horizontal line at y=0
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

# Set labels and title
ax.set_xlabel('Fitted Values', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residual vs Fitted Plot - Baseline Linear Regression', fontsize=16, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3)

# Add R² information
r2_lr = r2_score(y_train_numeric, y_pred_lr)
ax.text(0.05, 0.95, f'R² = {r2_lr:.3f}', transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add statistics
residual_stats = f'Mean Residual: {residuals.mean():.3f}\nStd Residual: {residuals.std():.3f}'
ax.text(0.05, 0.05, residual_stats, transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Save image
plt.tight_layout()
plt.savefig('fig_6_1b_outlier_influence.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 46. Create fig_6_1c_interaction_plot.png - 2D contour of two key predictors vs predicted life expectancy
print("Creating fig_6_1c_interaction_plot.png...")

# Select two key predictors (top 2 from correlation)
key_predictors = top_3_predictors[:2]
print(f"Key predictors for interaction plot: {key_predictors}")

# Prepare data for shallow tree
X_tree = X_train_diag[key_predictors].dropna()
y_tree = y_train_diag.loc[X_tree.index]

# Train shallow decision tree (max_depth=3)
tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_tree, y_tree)

# Create grid for contour plot
x1_min, x1_max = X_tree[key_predictors[0]].min(), X_tree[key_predictors[0]].max()
x2_min, x2_max = X_tree[key_predictors[1]].min(), X_tree[key_predictors[1]].max()

# Extend range slightly for better visualization
x1_range = x1_max - x1_min
x2_range = x2_max - x2_min
x1_min -= 0.1 * x1_range
x1_max += 0.1 * x1_range
x2_min -= 0.1 * x2_range
x2_max += 0.1 * x2_range

# Create grid
x1_grid = np.linspace(x1_min, x1_max, 100)
x2_grid = np.linspace(x2_min, x2_max, 100)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

# Flatten grid for prediction
grid_points = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

# Make predictions
Z_pred = tree_model.predict(grid_points)
Z_pred = Z_pred.reshape(X1_grid.shape)

# Create contour plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create contour plot
contour = ax.contourf(X1_grid, X2_grid, Z_pred, levels=20, cmap='viridis', alpha=0.8)
contour_lines = ax.contour(X1_grid, X2_grid, Z_pred, levels=10, colors='black', alpha=0.5, linewidths=0.5)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Predicted Life Expectancy (years)', rotation=270, labelpad=20)

# Overlay actual data points
ax.scatter(X_tree[key_predictors[0]], X_tree[key_predictors[1]], 
          c=y_tree, cmap='viridis', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)

# Set labels and title
ax.set_xlabel(key_predictors[0].strip(), fontsize=12)
ax.set_ylabel(key_predictors[1].strip(), fontsize=12)
ax.set_title(f'2D Contour Plot: {key_predictors[0].strip()} vs {key_predictors[1].strip()}\n(Predicted Life Expectancy using Shallow Tree)', 
             fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3)

# Add tree model info
tree_r2 = tree_model.score(X_tree, y_tree)
ax.text(0.02, 0.98, f'Tree R² = {tree_r2:.3f}\nMax Depth = 3', transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        verticalalignment='top')

# Save image
plt.tight_layout()
plt.savefig('fig_6_1c_interaction_plot.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 47. Summary of diagnostic plots
print("Diagnostic plots completed!")
print("Diagnostic Results:")
print(f"  - Top 3 predictors: {top_3_predictors}")
print(f"  - Linear regression R²: {r2_lr:.4f}")
print(f"  - Shallow tree R²: {tree_r2:.4f}")
print(f"  - Training samples used: {len(X_train_diag)}")
print("New generated files:")
print("- fig_6_1a_linearity_plot.png")
print("- fig_6_1b_outlier_influence.png")
print("- fig_6_1c_interaction_plot.png")

# 48. Define sklearn pipelines and param_grids for GridSearchCV
print("Defining sklearn pipelines and param_grids...")

# Import additional sklearn modules for pipelines
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# Define multiple pipelines for different algorithms
pipelines = {}

# 1. Linear Regression Pipeline
pipelines['linear'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', LinearRegression())
])

# 2. Ridge Regression Pipeline
pipelines['ridge'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', Ridge())
])

# 3. Lasso Regression Pipeline
pipelines['lasso'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', Lasso())
])

# 4. ElasticNet Pipeline
pipelines['elasticnet'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', ElasticNet())
])

# 5. Polynomial Features + Ridge Pipeline
pipelines['polynomial'] = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', Ridge())
])

# 6. PCA + Ridge Pipeline
pipelines['pca'] = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('regressor', Ridge())
])

# 7. Random Forest Pipeline
pipelines['random_forest'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 8. Gradient Boosting Pipeline
pipelines['gradient_boosting'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# 9. Support Vector Regression Pipeline
pipelines['svr'] = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', SVR())
])

# Define parameter grids for GridSearchCV
param_grids = {}

# Linear Regression parameters (reduced for faster training)
param_grids['linear'] = {
    'feature_selection__k': [20, 50, 100],
    'regressor__fit_intercept': [True]
}

# Ridge Regression parameters (reduced for faster training)
param_grids['ridge'] = {
    'feature_selection__k': [20, 50, 100],
    'regressor__alpha': [0.1, 1.0, 10.0],
    'regressor__fit_intercept': [True]
}

# Lasso Regression parameters (reduced for faster training)
param_grids['lasso'] = {
    'feature_selection__k': [20, 50, 100],
    'regressor__alpha': [0.01, 0.1, 1.0],
    'regressor__fit_intercept': [True]
}

# ElasticNet parameters (reduced for faster training)
param_grids['elasticnet'] = {
    'feature_selection__k': [20, 50, 100],
    'regressor__alpha': [0.01, 0.1, 1.0],
    'regressor__l1_ratio': [0.3, 0.5, 0.7],
    'regressor__fit_intercept': [True]
}

# Polynomial Features parameters (reduced for faster training)
param_grids['polynomial'] = {
    'poly_features__degree': [2],
    'poly_features__interaction_only': [True],
    'feature_selection__k': [50, 100],
    'regressor__alpha': [0.1, 1.0]
}

# PCA parameters (reduced for faster training)
param_grids['pca'] = {
    'pca__n_components': [10, 15, 20],
    'regressor__alpha': [0.1, 1.0]
}

# Random Forest parameters (reduced for faster training)
param_grids['random_forest'] = {
    'feature_selection__k': [50, 100],
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [10, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

# Gradient Boosting parameters (reduced for faster training)
param_grids['gradient_boosting'] = {
    'feature_selection__k': [50, 100],
    'regressor__n_estimators': [50, 100],
    'regressor__learning_rate': [0.1, 0.2],
    'regressor__max_depth': [3, 5],
    'regressor__subsample': [0.9, 1.0]
}

# Support Vector Regression parameters (reduced for faster training)
param_grids['svr'] = {
    'feature_selection__k': [20, 50],
    'regressor__C': [1.0, 10.0],
    'regressor__gamma': ['scale', 0.01],
    'regressor__kernel': ['rbf', 'linear']
}

# Create GridSearchCV objects (5-fold CV)
grid_searches = {}
for name, pipeline in pipelines.items():
    grid_searches[name] = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

print(f"Defined {len(pipelines)} pipelines with parameter grids:")
for name in pipelines.keys():
    print(f"  - {name}: {len(param_grids[name])} parameter combinations")

# 49. Create pipeline diagram
print("Creating pipeline diagram...")

fig, ax = plt.subplots(figsize=(20, 12))

# Define colors for different components
colors = {
    'preprocessing': '#E6F3FF',  # Light blue
    'feature_selection': '#FFE6E6',  # Light red
    'dimensionality': '#E6FFE6',  # Light green
    'regressor': '#FFF0E6',  # Light orange
    'ensemble': '#F0E6FF'  # Light purple
}

# Define pipeline components and their positions
pipeline_components = {
    'Data Input': (1, 10, 'preprocessing'),
    'StandardScaler': (3, 10, 'preprocessing'),
    'PolynomialFeatures': (5, 10, 'preprocessing'),
    'PCA': (5, 8, 'dimensionality'),
    'SelectKBest': (7, 10, 'feature_selection'),
    'LinearRegression': (9, 10, 'regressor'),
    'Ridge': (9, 8, 'regressor'),
    'Lasso': (9, 6, 'regressor'),
    'ElasticNet': (9, 4, 'regressor'),
    'RandomForest': (11, 10, 'ensemble'),
    'GradientBoosting': (11, 8, 'ensemble'),
    'SVR': (11, 6, 'ensemble'),
    'Predictions': (13, 8, 'preprocessing')
}

# Draw boxes for each component
for name, (x, y, category) in pipeline_components.items():
    # Create rectangle
    rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                        facecolor=colors[category], 
                        edgecolor='black', 
                        linewidth=1.5)
    ax.add_patch(rect)
    
    # Add text
    ax.text(x, y, name, ha='center', va='center', 
           fontsize=10, fontweight='bold')

# Draw arrows to show pipeline flow
arrows = [
    # Main preprocessing flow
    ((1, 10), (3, 10)),  # Data -> StandardScaler
    ((3, 10), (5, 10)),  # StandardScaler -> PolynomialFeatures
    ((5, 10), (7, 10)),  # PolynomialFeatures -> SelectKBest
    
    # Alternative paths
    ((3, 10), (5, 8)),   # StandardScaler -> PCA
    ((5, 8), (7, 10)),   # PCA -> SelectKBest (merge)
    
    # Feature selection to regressors
    ((7, 10), (9, 10)),  # SelectKBest -> LinearRegression
    ((7, 10), (9, 8)),   # SelectKBest -> Ridge
    ((7, 10), (9, 6)),   # SelectKBest -> Lasso
    ((7, 10), (9, 4)),   # SelectKBest -> ElasticNet
    
    # Regressors to ensemble methods
    ((9, 10), (11, 10)), # LinearRegression -> RandomForest
    ((9, 8), (11, 8)),   # Ridge -> GradientBoosting
    ((9, 6), (11, 6)),   # Lasso -> SVR
    
    # Final predictions
    ((11, 10), (13, 8)), # RandomForest -> Predictions
    ((11, 8), (13, 8)),  # GradientBoosting -> Predictions
    ((11, 6), (13, 8)),  # SVR -> Predictions
]

# Draw arrows
for (start_x, start_y), (end_x, end_y) in arrows:
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

# Add pipeline labels
pipeline_labels = [
    ('Linear Pipeline', 2, 12),
    ('Ridge Pipeline', 2, 11),
    ('Lasso Pipeline', 2, 9),
    ('ElasticNet Pipeline', 2, 7),
    ('Polynomial Pipeline', 6, 12),
    ('PCA Pipeline', 6, 9),
    ('Random Forest Pipeline', 10, 12),
    ('Gradient Boosting Pipeline', 10, 11),
    ('SVR Pipeline', 10, 9)
]

for label, x, y in pipeline_labels:
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=12, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['preprocessing'], label='Preprocessing'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['feature_selection'], label='Feature Selection'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['dimensionality'], label='Dimensionality Reduction'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['regressor'], label='Linear Regressors'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['ensemble'], label='Ensemble Methods')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

# Set title and labels
ax.set_title('Machine Learning Pipelines for Life Expectancy Prediction\n(GridSearchCV with 5-fold Cross-Validation)', 
             fontsize=16, fontweight='bold', pad=20)

# Set axis properties
ax.set_xlim(0, 14)
ax.set_ylim(2, 13)
ax.set_aspect('equal')
ax.axis('off')

# Add grid search information
info_text = """GridSearchCV Configuration:
• 5-fold Cross-Validation
• Scoring: R²
• Parallel Processing: n_jobs=-1
• Total Pipelines: 9
• Parameter Combinations: ~50-200 per pipeline"""
ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        verticalalignment='bottom')

# Save image
plt.tight_layout()
plt.savefig('fig_6_2_pipelines.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 50. Summary
print("Pipeline definition and diagram completed!")
print("Pipeline Summary:")
print(f"  - Total pipelines defined: {len(pipelines)}")
print(f"  - GridSearchCV objects created: {len(grid_searches)}")
print(f"  - Cross-validation folds: 5")
print(f"  - Scoring metric: R²")
print("Pipeline types:")
for name in pipelines.keys():
    print(f"  - {name}")
print("New generated files:")
print("- fig_6_2_pipelines.png")

# 51. Execute training/tuning for all models
print("Executing training/tuning for all models...")

# Import joblib for model persistence
import joblib
from sklearn.metrics import mean_squared_error
import time

# Prepare data for training (use processed data)
X_train_final = X_train_processed
X_test_final = X_test_processed
y_train_final = y_train_processed
y_test_final = y_test_processed

print(f"Training data shape: {X_train_final.shape}")
print(f"Test data shape: {X_test_final.shape}")

# Store results
cv_results = {}
best_models = {}
training_times = {}

# Train each model with GridSearchCV (prioritize key models)
priority_models = ['ridge', 'random_forest', 'gradient_boosting', 'linear', 'lasso']
all_models = list(grid_searches.keys())

# Train priority models first
for name in priority_models:
    if name in grid_searches:
        print(f"\nTraining {name} model...")
        start_time = time.time()
        
        try:
            # Fit the grid search
            grid_search = grid_searches[name]
            grid_search.fit(X_train_final, y_train_final)
            
            # Store results
            cv_results[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_scores': grid_search.cv_results_['mean_test_score'],
                'cv_std': grid_search.cv_results_['std_test_score'],
                'rmse_scores': np.sqrt(-grid_search.cv_results_['mean_test_score']),  # Convert R² to RMSE
                'rmse_std': np.sqrt(grid_search.cv_results_['std_test_score'])
            }
            
            best_models[name] = grid_search.best_estimator_
            training_times[name] = time.time() - start_time
            
            print(f"  Best CV R²: {grid_search.best_score_:.4f}")
            print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
            print(f"  Training time: {training_times[name]:.2f} seconds")
            
        except Exception as e:
            print(f"  Error training {name}: {str(e)}")
            cv_results[name] = None
            best_models[name] = None
            training_times[name] = None

# Train remaining models if time permits
remaining_models = [name for name in all_models if name not in priority_models]
for name in remaining_models:
    print(f"\nTraining {name} model...")
    start_time = time.time()
    
    try:
        # Fit the grid search
        grid_search = grid_searches[name]
        grid_search.fit(X_train_final, y_train_final)
        
        # Store results
        cv_results[name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_scores': grid_search.cv_results_['mean_test_score'],
            'cv_std': grid_search.cv_results_['std_test_score'],
            'rmse_scores': np.sqrt(-grid_search.cv_results_['mean_test_score']),  # Convert R² to RMSE
            'rmse_std': np.sqrt(grid_search.cv_results_['std_test_score'])
        }
        
        best_models[name] = grid_search.best_estimator_
        training_times[name] = time.time() - start_time
        
        print(f"  Best CV R²: {grid_search.best_score_:.4f}")
        print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        print(f"  Training time: {training_times[name]:.2f} seconds")
        
    except Exception as e:
        print(f"  Error training {name}: {str(e)}")
        cv_results[name] = None
        best_models[name] = None
        training_times[name] = None

# Find the best model overall
valid_results = {k: v for k, v in cv_results.items() if v is not None}
if valid_results:
    best_model_name = max(valid_results.keys(), key=lambda x: valid_results[x]['best_score'])
    best_model = best_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best CV R²: {valid_results[best_model_name]['best_score']:.4f}")
    print(f"Best CV RMSE: {np.sqrt(-valid_results[best_model_name]['best_score']):.4f}")
    print(f"Best parameters: {valid_results[best_model_name]['best_params']}")
    
    # Save best model
    joblib.dump(best_model, 'best_model.joblib')
    print("Best model saved as 'best_model.joblib'")
else:
    print("No models were successfully trained!")
    best_model_name = None
    best_model = None

# 52. Create fig_6_3a_cv_scores.png - Bar chart of mean CV RMSE by model
print("Creating fig_6_3a_cv_scores.png...")

# Prepare data for plotting
model_names = []
rmse_means = []
rmse_stds = []
r2_scores = []

for name, results in valid_results.items():
    model_names.append(name.replace('_', ' ').title())
    rmse_means.append(results['rmse_scores'].mean())
    rmse_stds.append(results['rmse_std'].mean())
    r2_scores.append(results['best_score'])

# Create bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
bars = ax.bar(model_names, rmse_means, yerr=rmse_stds, 
              capsize=5, alpha=0.7, color='skyblue', edgecolor='black')

# Highlight the best model
if best_model_name:
    best_idx = model_names.index(best_model_name.replace('_', ' ').title())
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

# Set labels and title
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Mean CV RMSE', fontsize=12)
ax.set_title('Cross-Validation RMSE Scores by Model\n(5-fold CV with Error Bars)', 
             fontsize=16, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for i, (bar, mean_val, std_val, r2_val) in enumerate(zip(bars, rmse_means, rmse_stds, r2_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.1,
            f'RMSE: {mean_val:.3f}\nR²: {r2_val:.3f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Add legend
if best_model_name:
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='gold', edgecolor='red', linewidth=2, label='Best Model'),
        plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', edgecolor='black', label='Other Models')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

# Save image
plt.tight_layout()
plt.savefig('fig_6_3a_cv_scores.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 53. Create fig_6_3b_hyperparam_trace.png - Hyperparameter trace for winning model
print("Creating fig_6_3b_hyperparam_trace.png...")

if best_model_name and best_model_name in grid_searches:
    grid_search_best = grid_searches[best_model_name]
    best_results = cv_results[best_model_name]
    
    # Get the key hyperparameter (most important one)
    param_names = list(best_results['best_params'].keys())
    
    # Choose the most important parameter for visualization
    # Priority: regularization parameters, then feature selection, then others
    key_param = None
    for param in param_names:
        if 'alpha' in param or 'C' in param:
            key_param = param
            break
    if not key_param:
        for param in param_names:
            if 'k' in param or 'n_components' in param:
                key_param = param
                break
    if not key_param and param_names:
        key_param = param_names[0]
    
    if key_param:
        # Extract parameter values and corresponding scores
        param_values = grid_search_best.cv_results_[f'param_{key_param}']
        rmse_scores = np.sqrt(-grid_search_best.cv_results_['mean_test_score'])
        rmse_stds = np.sqrt(grid_search_best.cv_results_['std_test_score'])
        
        # Convert parameter values to numeric if possible
        try:
            param_values_numeric = [float(str(v)) for v in param_values]
        except:
            param_values_numeric = list(range(len(param_values)))
            param_labels = [str(v) for v in param_values]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create line plot with error bars
        if len(set(param_values_numeric)) > 1:
            # Sort by parameter value for line plot
            sorted_indices = np.argsort(param_values_numeric)
            sorted_params = np.array(param_values_numeric)[sorted_indices]
            sorted_rmse = np.array(rmse_scores)[sorted_indices]
            sorted_std = np.array(rmse_stds)[sorted_indices]
            
            # Plot line with error bars
            ax.errorbar(sorted_params, sorted_rmse, yerr=sorted_std, 
                       marker='o', linewidth=2, markersize=8, capsize=5,
                       color='blue', alpha=0.7, label='CV RMSE')
            
            # Highlight best point
            best_param_value = best_results['best_params'][key_param]
            best_rmse = np.sqrt(-best_results['best_score'])
            ax.scatter([best_param_value], [best_rmse], 
                      color='red', s=100, zorder=5, label='Best Parameter')
            
            ax.set_xlabel(f'{key_param.replace("__", " → ")}', fontsize=12)
        else:
            # Bar plot for categorical parameters
            bars = ax.bar(range(len(param_values_numeric)), rmse_scores, 
                         yerr=rmse_stds, capsize=5, alpha=0.7, color='skyblue')
            
            # Highlight best bar
            best_idx = list(param_values).index(best_results['best_params'][key_param])
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            ax.set_xlabel(f'{key_param.replace("__", " → ")}', fontsize=12)
            ax.set_xticks(range(len(param_values_numeric)))
            ax.set_xticklabels([str(v) for v in param_values], rotation=45)
        
        ax.set_ylabel('CV RMSE', fontsize=12)
        ax.set_title(f'Hyperparameter Tuning Trace: {best_model_name.replace("_", " ").title()}\n{key_param.replace("__", " → ")} vs CV RMSE', 
                     fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Add best parameter info
        info_text = f'Best {key_param}: {best_results["best_params"][key_param]}\nBest CV RMSE: {np.sqrt(-best_results["best_score"]):.4f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
        
        # Save image
        plt.tight_layout()
        plt.savefig('fig_6_3b_hyperparam_trace.png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Hyperparameter trace created for {key_param}")
    else:
        print("No suitable hyperparameter found for trace plot")
else:
    print("No best model available for hyperparameter trace")

# 54. Final summary
print("\nTraining and tuning completed!")
print("Results Summary:")
print(f"  - Models trained: {len(valid_results)}")
print(f"  - Best model: {best_model_name}")
if best_model_name:
    print(f"  - Best CV R²: {valid_results[best_model_name]['best_score']:.4f}")
    print(f"  - Best CV RMSE: {np.sqrt(-valid_results[best_model_name]['best_score']):.4f}")
    print(f"  - Training time: {training_times[best_model_name]:.2f} seconds")

print("\nModel Performance Ranking (by CV R²):")
sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['best_score'], reverse=True)
for i, (name, results) in enumerate(sorted_models, 1):
    print(f"  {i}. {name}: R² = {results['best_score']:.4f}, RMSE = {np.sqrt(-results['best_score']):.4f}")

print("\nNew generated files:")
print("- fig_6_3a_cv_scores.png")
print("- fig_6_3b_hyperparam_trace.png")
print("- best_model.joblib")

# 55. Create fig_7_1_design.png - Data flow schematic
print("Creating fig_7_1_design.png - Data flow schematic...")

fig, ax = plt.subplots(figsize=(20, 8))

# Define colors for different stages
colors = {
    'data': '#E6F3FF',      # Light blue
    'preprocessing': '#FFE6E6',  # Light red
    'engineering': '#E6FFE6',   # Light green
    'splitting': '#FFF0E6',     # Light orange
    'validation': '#F0E6FF',    # Light purple
    'selection': '#FFE6F0',     # Light pink
    'testing': '#E6F0FF'        # Light cyan
}

# Define data flow components and their positions
flow_components = {
    'Raw Data': (1, 4, 'data'),
    'Clean': (3, 4, 'preprocessing'),
    'Engineer': (5, 4, 'engineering'),
    'Split (80/20)': (7, 4, 'splitting'),
    'CV (5-fold)': (9, 4, 'validation'),
    'Select': (11, 4, 'selection'),
    'Test': (13, 4, 'testing')
}

# Draw boxes for each component
for name, (x, y, category) in flow_components.items():
    # Create rectangle
    rect = plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8, 
                        facecolor=colors[category], 
                        edgecolor='black', 
                        linewidth=2)
    ax.add_patch(rect)
    
    # Add text
    ax.text(x, y, name, ha='center', va='center', 
           fontsize=12, fontweight='bold')

# Draw arrows to show data flow
arrows = [
    ((1.6, 4), (2.4, 4)),  # Raw Data → Clean
    ((3.6, 4), (4.4, 4)),  # Clean → Engineer
    ((5.6, 4), (6.4, 4)),  # Engineer → Split
    ((7.6, 4), (8.4, 4)),  # Split → CV
    ((9.6, 4), (10.4, 4)), # CV → Select
    ((11.6, 4), (12.4, 4)) # Select → Test
]

# Draw arrows
for (start_x, start_y), (end_x, end_y) in arrows:
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
               arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))

# Add detailed descriptions for each stage
stage_descriptions = {
    'Raw Data': 'Life Expectancy Data.csv\n(2,940 records, 22 features)',
    'Clean': 'Missing value imputation\nOutlier capping\nData type conversion',
    'Engineer': 'Feature creation\nCorrelation analysis\nRegion join',
    'Split (80/20)': 'Train: 2,352 samples\nTest: 588 samples\nStratified by Year',
    'CV (5-fold)': 'GridSearchCV\n9 algorithms\n5-fold cross-validation',
    'Select': 'Best model selection\nHyperparameter tuning\nPerformance evaluation',
    'Test': 'Final evaluation\nModel deployment\nPerformance metrics'
}

# Add descriptions below each stage
for name, (x, y, category) in flow_components.items():
    description = stage_descriptions[name]
    ax.text(x, y-1.2, description, ha='center', va='center', 
           fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8))

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['data'], label='Raw Data'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['preprocessing'], label='Preprocessing'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['engineering'], label='Feature Engineering'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['splitting'], label='Data Splitting'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['validation'], label='Cross-Validation'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['selection'], label='Model Selection'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['testing'], label='Final Testing')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

# Set title
ax.set_title('Data Flow Schematic: Life Expectancy Prediction Pipeline\nFrom Raw Data to Final Model Testing', 
             fontsize=18, fontweight='bold', pad=30)

# Set axis properties
ax.set_xlim(0, 14)
ax.set_ylim(1, 6)
ax.set_aspect('equal')
ax.axis('off')

# Add pipeline summary information
summary_text = """Pipeline Summary:
• Total Records: 2,940
• Features: 22 → 30+ (after engineering)
• Train/Test Split: 80/20
• Cross-Validation: 5-fold
• Algorithms Tested: 9
• Best Model: Selected via GridSearchCV
• Final Evaluation: Test set performance"""
ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        verticalalignment='bottom')

# Save image
plt.tight_layout()
plt.savefig('fig_7_1_design.png', dpi=100, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Data flow schematic completed!")
print("New generated files:")
print("- fig_7_1_design.png")