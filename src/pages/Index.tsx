
import { useState } from 'react';
import { Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import NavigationHeader from '@/components/NavigationHeader';
import CodeBlock from '@/components/CodeBlock';
import SectionBlock from '@/components/SectionBlock';
import GraphSection from '@/components/GraphSection';

const Index = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const { toast } = useToast();

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  const preprocessingCode = `import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# --- Step 1: Load the dataset ---
# This code assumes '21mic7040_dataset.xlsx' is in the same directory as this script.
try:
    df = pd.read_excel('21mic7040_dataset.xlsx')
    print("Dataset '21mic7040_dataset.xlsx' loaded successfully.")
except FileNotFoundError:
    print("Error: '21mic7040_dataset.xlsx' not found.")
    print("Please ensure the dataset file is in the same directory as this script.")
    exit() # Exit the script if the file is not found

print("\\nOriginal DataFrame Head:")
print(df.head())
print("\\nOriginal DataFrame Info:")
print(df.info())

# 2. Feature Selection and Renaming
# Define a mapping for clearer column names
column_mapping = {
    'Q1. What platform do you usually play Genshin Impact on?': 'platform',
    'Q2. Which region in the game is your favorite?': 'region',
    'Q3. What is your favorite element?': 'element',
    'Q4. What is your favorite weapon type?': 'weapon',
    'Q5. Who is your favorite 5-star character?': 'fav_5star',
    'Q6. Which 5 star character do you hate the most': 'hate_5star',
    'Q7. Who is your favourite archon?': 'fav_archon',
    'Q8. Do you play spiral abyss?': 'spiral_abyss',
    'Q9. What is your adventure rank?': 'adventure_rank'
}

# Select only the relevant columns and rename them
df_preprocessed = df[['user', 'Timestamp'] + list(column_mapping.keys())].rename(columns=column_mapping)

# 3. Data Cleaning and Consistency for Categorical Columns
# Identify categorical columns (excluding 'user' and 'Timestamp')
categorical_cols = [col for col in df_preprocessed.columns if df_preprocessed[col].dtype == 'object' and col not in ['user', 'Timestamp']]

for col in categorical_cols:
    # Convert to lowercase and strip whitespace
    df_preprocessed[col] = df_preprocessed[col].str.lower().str.strip()

# Specific typo correction for 'platform'
df_preprocessed['platform'] = df_preprocessed['platform'].replace({'mocile': 'mobile', 'movile': 'mobile'})

# 4. Handle Missing Values (Check only)
print("\\nMissing values before imputation:")
print(df_preprocessed.isnull().sum())

# 5. Numerical Feature Preprocessing: 'adventure_rank'
# Convert 'adventure_rank' to numeric, coercing errors will turn non-numeric into NaN
df_preprocessed['adventure_rank'] = pd.to_numeric(df_preprocessed['adventure_rank'], errors='coerce')

# If any NaNs were created due to coercion (e.g., non-numeric strings), handle them.
if df_preprocessed['adventure_rank'].isnull().any():
    print("\\nWarning: Non-numeric values found and converted to NaN in 'adventure_rank'. Imputing with median.")
    df_preprocessed['adventure_rank'].fillna(df_preprocessed['adventure_rank'].median(), inplace=True)

# Scale 'adventure_rank'
scaler = StandardScaler()
df_preprocessed['adventure_rank_scaled'] = scaler.fit_transform(df_preprocessed[['adventure_rank']])

# 6. One-Hot Encoding for Categorical Features
# Identify categorical features for encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df_preprocessed[categorical_cols])

# Create a DataFrame from encoded features with proper column names
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_preprocessed.index)

# 7. Combine Preprocessed Data
# Drop original categorical columns and the non-scaled adventure_rank
df_final_preprocessed = df_preprocessed.drop(columns=categorical_cols + ['adventure_rank'])

# Concatenate the encoded categorical features and scaled numerical feature
df_final_preprocessed = pd.concat([df_final_preprocessed, df_encoded], axis=1)

print("\\nPreprocessed DataFrame Head:")
print(df_final_preprocessed.head())

print("\\nPreprocessed DataFrame Info:")
print(df_final_preprocessed.info())

df_final_preprocessed.to_excel('preprocessed_21mic7040_dataset.xlsx', index=False)
print("\\nPreprocessed data saved to 'preprocessed_21mic7040_dataset.xlsx'")`;

  const clusteringCode = `import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans # KMeans is needed to get the cluster labels

# Load the dataset
df = pd.read_excel('dataset.xlsx')

clustering_cols = [col for col in df.columns if col.startswith('Q')]
df_clustering = df[clustering_cols].copy()
categorical_cols = df_clustering.columns

# Apply One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df_clustering[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(encoded_df)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(encoded_df)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df, s=100, alpha=0.8)
plt.title('2D PCA of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('cluster_scatterplot_from_xlsx.png') # Saves the plot to a file
plt.close() # Closes the plot to prevent it from displaying directly in some environments`;

  const platformChartCode = `import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel('dataset.xlsx')

df['Q1. What platform do you usually play Genshin Impact on?'] = df['Q1. What platform do you usually play Genshin Impact on?'].replace({
    'mocile': 'mobile',
    'movile': 'mobile'
})

platform_data_corrected = df['Q1. What platform do you usually play Genshin Impact on?']

plt.figure(figsize=(10, 6))
sns.countplot(y=platform_data_corrected, order=platform_data_corrected.value_counts().index, palette='viridis')
plt.title('Most Popular Platforms for Playing Genshin Impact (Corrected)')
plt.xlabel('Number of Respondents')
plt.ylabel('Platform')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('most_popular_platforms_bar_chart_corrected.png')
plt.close()

print("Typos corrected and updated bar chart for 'Most Popular Platforms' generated successfully.")`;

  const regionArchonCode = `import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('dataset.xlsx')

df['Q1. What platform do you usually play Genshin Impact on?'] = df['Q1. What platform do you usually play Genshin Impact on?']
.replace({
    'mocile': 'mobile',
    'movile': 'mobile'
})

region_col = 'Q2. Which region in the game is your favorite?'
archon_col = 'Q7. Who is your favourite archon?'

# Create a grouped bar chart
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x=region_col, hue=archon_col, palette='tab10')

plt.title('Region Preference by Favourite Archon')
plt.xlabel('Favourite Region')
plt.ylabel('Number of Respondents')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
plt.legend(title='Favourite Archon', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig('region_archon_preference_grouped_bar_chart.png') # Saving to a general name
plt.close()

print("Grouped bar chart for Region and Archon Preference generated successfully.")`;

  const predictionCode = `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load preprocessed dataset
preprocessed_file_name = 'preprocessed_21mic7040_dataset.xlsx'

try:
    df_preprocessed = pd.read_excel(preprocessed_file_name)
    print(f"Dataset '{preprocessed_file_name}' loaded successfully.")

    # Define target variable (platform) from one-hot encoded columns
    platform_cols = [col for col in df_preprocessed.columns if col.startswith('platform_')]
    if not platform_cols:
        raise ValueError("Platform one-hot encoded columns not found. Check your preprocessing.")

    y = df_preprocessed[platform_cols].idxmax(axis=1).str.replace('platform_', '')

    # Define features (exclude user, timestamp, platform, and unscaled rank if present)
    features_to_exclude = ['user', 'Timestamp'] + platform_cols
    if 'adventure_rank' in df_preprocessed.columns:
        features_to_exclude.append('adventure_rank')

    X = df_preprocessed.drop(columns=features_to_exclude, errors='ignore')

    if X.empty:
        raise ValueError("No usable features found in dataset.")

    non_numeric_cols_X = X.select_dtypes(include=['object', 'category']).columns
    if not non_numeric_cols_X.empty:
        print(f"Warning: Non-numeric columns in features: {list(non_numeric_cols_X)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\\n--- Classification Results ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"This means the model correctly predicted the platform for {accuracy:.2%} of the test set.")

except FileNotFoundError:
    print(f"Error: File '{preprocessed_file_name}' not found.")
except ValueError as ve:
    print(f"Data Error: {ve}")
except Exception as e:
    print(f"Unexpected error: {e}")`;

  return (
    <div className={`min-h-screen ${isDarkMode ? 'dark bg-slate-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <NavigationHeader 
        isDarkMode={isDarkMode} 
        toggleTheme={toggleTheme} 
      />
      
      <main className="container mx-auto px-6 py-8 mt-20 max-w-7xl">
        {/* Header Section */}
<div className="text-center mb-16">
  <Badge variant="secondary" className="mb-4 text-sm bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-sm">
    Data Analytics Report
  </Badge>
  <h1 className="text-5xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-slate-800 to-slate-600 dark:from-white dark:to-slate-300 drop-shadow-sm">
    Genshin Impact Survey Analysis
  </h1>
  <p className="text-xl text-gray-700 dark:text-gray-300 max-w-3xl mx-auto px-4 sm:px-8 leading-relaxed">
    <span className="font-medium text-slate-800 dark:text-slate-100">
      Player Behavior & Preferences
    </span>{" "}
    Analysis with{" "}
    <span className="font-medium text-slate-700 dark:text-slate-200">
      Machine Learning Insights
    </span>
  </p>
</div>


        {/* Problem Statement */}
        <div className="mb-12">
          <SectionBlock 
            title="Problem Statement" 
            variant="primary"
            className="text-center"
          >
            <p className="text-lg font-medium">
              "Can we predict whether a player participates in Spiral Abyss based on their gameplay preferences and profile?"
            </p>
          </SectionBlock>
        </div>

        {/* Step 1: Preprocessing */}
        <Card className="mb-12 border-2">
          <CardHeader className="bg-slate-100 dark:bg-slate-800">
            <CardTitle className="text-2xl font-bold flex items-center gap-3">
              <Badge className="bg-blue-500 text-white">STEP 1</Badge>
              <span>Data Preprocessing</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8">
            <CodeBlock 
              code={preprocessingCode}
              title="Python Code: Data Preprocessing Pipeline"
              language="python"
            />

            <SectionBlock 
              title="Data Preprocessing Summary" 
              variant="accent"
              className="mt-8"
            >
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3 text-blue-600">Key Steps:</h4>
                  <ul className="space-y-2 text-sm">
                    <li><strong>Dataset Loading:</strong> Loads the excel file with error handling</li>
                    <li><strong>Column Renaming:</strong> Renames survey columns for readability</li>
                    <li><strong>Data Cleaning:</strong> Converts text to lowercase, fixes typos</li>
                    <li><strong>Missing Values:</strong> Handles nulls with median imputation</li>
                    <li><strong>Scaling:</strong> Standardizes adventure_rank column</li>
                    <li><strong>Encoding:</strong> One-hot encodes categorical variables</li>
                  </ul>
                </div>
                <div className="flex items-center justify-center">
                  <Button className="bg-green-600 hover:bg-green-700 text-white">
                    <Download className="w-4 h-4 mr-2" />
                    Download Preprocessed Dataset
                  </Button>
                </div>
              </div>
            </SectionBlock>
          </CardContent>
        </Card>

        {/* Step 2: Representation */}
        <Card className="mb-12 border-2">
          <CardHeader className="bg-slate-100 dark:bg-slate-800">
            <CardTitle className="text-2xl font-bold flex items-center gap-3">
              <Badge className="bg-purple-500 text-white">STEP 2</Badge>
              <span>Data Representation</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8">
            {/* Clustering Analysis */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4">Clustering Analysis</h3>
              <CodeBlock 
                code={clusteringCode}
                title="Python Code: K-Means Clustering with PCA Visualization"
                language="python"
              />
              
              <GraphSection 
      title="Cluster Analysis Visualization" 
      imageSrc="/cluster_scatterplot_from_xlsx.png" 
      imageAlt="PCA Cluster Analysis Graph" 
       />


              {/* Cluster Insights */}
              <div className="grid md:grid-cols-3 gap-6 mt-8">
                <SectionBlock title="Cluster 0 Insights" variant="accent">
                  <ul className="space-y-2 text-sm">
                    <li>• Predominantly mobile players (100.00%)</li>
                    <li>• Top favorite region is Inazuma (25.00%)</li>
                    <li>• Top favorite element is Hydro (31.25%)</li>
                    <li>• Top favorite weapon type is Bow (31.25%)</li>
                    <li>• Most play Spiral Abyss (100.00%)</li>
                    <li>• Adventure Rank of 60 (62.50%)</li>
                    <li>• Top favorite Archon is Venti (31.25%)</li>
                  </ul>
                </SectionBlock>

                <SectionBlock title="Cluster 1 Insights" variant="accent">
                  <ul className="space-y-2 text-sm">
                    <li>• Mainly PC players (94.44%)</li>
                    <li>• Strong preference for Inazuma (33.33%)</li>
                    <li>• Top favorite element is Dendro (27.78%)</li>
                    <li>• Top favorite weapon is Polearm (38.89%)</li>
                    <li>• Most play Spiral Abyss (83.33%)</li>
                    <li>• Adventure Rank of 60 (83.33%)</li>
                    <li>• Top favorite Archon is Furina (33.33%)</li>
                  </ul>
                </SectionBlock>

                <SectionBlock title="Cluster 2 Insights" variant="accent">
                  <ul className="space-y-2 text-sm">
                    <li>• Mixed platform preference (37.50% mobile)</li>
                    <li>• Clear preference for Sumeru (43.75%)</li>
                    <li>• Top favorite element is Pyro (25.00%)</li>
                    <li>• Top favorite weapon is Bow (37.50%)</li>
                    <li>• <strong>None play Spiral Abyss (100% 'no')</strong></li>
                    <li>• Adventure Rank often 60 (37.50%)</li>
                    <li>• Top favorite Archon is Nahida (25.00%)</li>
                  </ul>
                </SectionBlock>
              </div>
            </div>

            <Separator className="my-8" />

            {/* Platform Analysis */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4">Platform Preference Analysis</h3>
              <CodeBlock 
                code={platformChartCode}
                title="Python Code: Platform Preference Bar Chart"
                language="python"
              />
              
              <GraphSection 
                title="Platform Preference Distribution"
                imageSrc="/bar.png"
                imageAlt="Platform Preference Graph"
              />

              <SectionBlock title="Platform Insights" variant="accent" className="mt-6">
                <h4 className="font-semibold mb-3">Key Findings:</h4>
                <ul className="space-y-3">
                  <li><strong>PC Dominance:</strong> PC is the most preferred platform among surveyed players</li>
                  <li><strong>Strong Mobile Presence:</strong> Mobile devices represent the second most popular platform</li>
                  <li><strong>Limited Console Usage:</strong> PlayStation usage is notably lower than PC and mobile</li>
                </ul>
              </SectionBlock>
            </div>

            <Separator className="my-8" />

            {/* Region vs Archon Analysis */}
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4">Region vs. Archon Preference Analysis</h3>
              <CodeBlock 
                code={regionArchonCode}
                title="Python Code: Grouped Bar Chart Analysis"
                language="python"
              />
              
              <GraphSection 
                title="Region vs. Archon Preference Correlation"
                imageSrc="/groupedbar.png"
                imageAlt="Region vs. Archon Preference Graph"
              />

              <SectionBlock title="Regional Alignment Analysis" variant="accent" className="mt-6">
                <h4 className="font-semibold mb-3">Key Findings:</h4>
                <ul className="space-y-3">
                  <li><strong>Strong Regional Alignment:</strong> Players overwhelmingly prefer Archons from their favorite regions</li>
                  <li><strong>Mondstadt:</strong> Venti dominates among Mondstadt enthusiasts</li>
                  <li><strong>Liyue:</strong> Zhongli is the clear favorite for Liyue players</li>
                  <li><strong>Inazuma:</strong> Raiden Shogun leads among Inazuma fans</li>
                  <li><strong>Sumeru:</strong> Nahida is most preferred by Sumeru region lovers</li>
                  <li><strong>Fontaine:</strong> Furina dominates Fontaine preferences</li>
                  <li><strong>Character Affinity:</strong> Strong correlation suggests deep lore connections</li>
                </ul>
              </SectionBlock>
            </div>
          </CardContent>
        </Card>

        {/* Machine Learning Prediction */}
        <Card className="mb-12 border-2">
          <CardHeader className="bg-slate-100 dark:bg-slate-800">
            <CardTitle className="text-2xl font-bold flex items-center gap-3">
              <Badge className="bg-green-500 text-white">STEP 3</Badge>
              <span>Find Accuracy</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8">
            <CodeBlock 
              code={predictionCode}
              title="Python Code: Random Forest Classifier for Platform Prediction"
              language="python"
            />

            <SectionBlock title="Random Forest Classifier Results" variant="primary" className="mt-8">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">Key Finding:</h4>
                  <p className="text-sm mb-4">
                    A Random Forest Classifier achieved <span className="font-bold text-lg">60% accuracy</span> in predicting gaming platform based on player preferences.
                  </p>
                  
                  <h4 className="font-semibold mb-3">Interpretation:</h4>
                  <p className="text-sm">
                    The model correctly predicts platform choice in 6 out of 10 cases, indicating moderate correlation between 
                    in-game preferences and platform selection.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-3">Business Impact:</h4>
                  <ul className="space-y-2 text-sm">
                    <li><strong>Targeted Marketing:</strong> Platform-specific campaigns for higher ROI</li>
                    <li><strong>Content Development:</strong> Tailored updates based on platform preferences</li>
                    <li><strong>Resource Allocation:</strong> Efficient development resource distribution</li>
                    <li><strong>Monetization:</strong> Platform-specific pricing strategies</li>
                    <li><strong>User Experience:</strong> Optimized UI/UX for predicted platforms</li>
                  </ul>
                </div>
              </div>
            </SectionBlock>
          </CardContent>
          <br></br>
        </Card>
         <SectionBlock
  title="Accuracy of Datasets"
  variant="accent"
  className="mt-6 text-center bg-green-100 text-green-800 border border-green-300 rounded-xl"
>
  <div className="flex justify-center">
    <img src="/barbar.png" alt="Accuracy Bar Graph" className="max-w-full h-auto" />
  </div>
</SectionBlock>
        {/* Footer */}
        <div className="text-center py-8 border-t">
          <p className="text-slate-600 dark:text-slate-400">
            © 2024 Genshin Impact Survey Analysis | Student ID: 21MIC7040 | Chandan Sathvik
          </p>
        </div>
      </main>
    </div>
  );
};

export default Index;
