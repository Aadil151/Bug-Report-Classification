import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

# Set random seeds for reproducibility
torch.manual_seed(39)
np.random.seed(39)
class BugReportDataset(Dataset):
    # Custom Dataset class for bug report data
    def __init__(self, texts, labels, tokenizer, max_length=512):
        #convert to list if not already
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        #convert text into BERT format
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    # BERT-based classifier for bug report classification
    def __init__(self, bert_model, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1) #to reduce overfitting
        self.classifier = nn.Linear(768, num_classes) #sets to binary classification


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) #get BERT outputs
        pooled_output = outputs[0][:, 0, :]#extract token representation
        pooled_output = self.dropout(pooled_output)#apply dropout for regularisation
        return self.classifier(pooled_output)

def train_epoch(model, dataloader, optimizer, device):
    # Trains the model for one epoch
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss() #cross entropy loss for classification
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)#get attention mask
        labels = batch['label'].to(device)#get labels

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask) #run through model
        loss = criterion(outputs, labels) #calculate loss
        loss.backward() #backpropagate
        optimizer.step()#update weights
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    # Evaluates the model on test data
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            outputs = model(input_ids, attention_mask) #run through model
            _, preds = torch.max(outputs, dim=1) #get predictions
            predictions.extend(preds.cpu().tolist()) #store predictions
            true_labels.extend(labels.tolist()) #store true labels

    return predictions, true_labels

def process_project(project_name, tokenizer, bert_model, device, batch_size=1, epochs=10, learning_rate=2e-5): #hyperparameters chosen based on device and optimal performance
    #priocess project
    print(f"\n Processing {project_name} ")
    
    path = f'{project_name}.csv'
    if not os.path.exists(path):
        print(f"file {path} not found, skipping project.")
        return None
    
    # Load and pre-process data
    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)

    # Merge Title and Body into a single column; if Body is NaN, use Title only
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Prepare data
    data = pd_all[['Unnamed: 0', 'Number', 'class', 'Title+Body']].rename(
        columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"}
    )

    # Split data
    X = data['text'].values
    y = data['sentiment'].values
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

    # Create datasets and dataloaders
    train_dataset = BugReportDataset(train_texts, train_labels, tokenizer)
    test_dataset = BugReportDataset(test_texts, test_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialise model and optimizer
    model = BERTClassifier(bert_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #AdamW optimizer

    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}')

    # Evaluation
    predictions, true_labels = evaluate(model, test_dataloader, device)

    # Calculate metrics
    metrics = {
        'Project': project_name,
        'Accuracy': accuracy_score(true_labels, predictions),
        'Precision': precision_score(true_labels, predictions, average='macro'),
        'Recall': recall_score(true_labels, predictions, average='macro'),
        'F1': f1_score(true_labels, predictions, average='macro')
    }

    # Print project results
    print(f"\n=== {project_name} BERT Classification Results ===")
    for metric, value in metrics.items():
        if metric != 'Project':
            print(f"{metric}: {value:.4f}")

    # Save individual project result
    pd.DataFrame([metrics]).to_csv(f'./{project_name}_BERT.csv', index=False)
    
    return metrics

def main():
    # List of projects to process
    projects = ['caffe', 'incubator-mxnet', 'tensorflow', 'pytorch', 'keras']
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    
    # Process each project
    all_results = []
    for project in projects:
        metrics = process_project(project, tokenizer, bert_model, device)
        if metrics:
            all_results.append(metrics)
    
    # Combine results
    if all_results:
        combined_results = pd.DataFrame(all_results)
        
        # Calculate average metrics
        avg_metrics = {
            'Project': 'AVERAGE',
            'Accuracy': combined_results['Accuracy'].mean(),
            'Precision': combined_results['Precision'].mean(),
            'Recall': combined_results['Recall'].mean(),
            'F1': combined_results['F1'].mean()
        }
        
        # Add average to results
        combined_results = pd.concat([combined_results, pd.DataFrame([avg_metrics])], ignore_index=True)
        
        # Save combined results
        combined_results.to_csv('./BERT_all_results.csv', index=False)
        
        # Print combined results
        print("combined results from BERT:")
        print(combined_results.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
        print("\nCombined results saved to: ./BERT_all_results.csv")
    else:
        print("No results were generated. Please check that the project CSV files exist.")

if __name__ == "__main__":
    main()