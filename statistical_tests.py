import pandas as pd
from scipy import stats

# Load the data
nb_results = pd.read_csv('NB_all_results.csv')
bert_results = pd.read_csv('BERT_all_results.csv')

# Remove the average row from both datasets
nb_results = nb_results[nb_results['Project'] != 'AVERAGE']
bert_results = bert_results[bert_results['Project'] != 'AVERAGE']

# Ensure the projects are in the same order
nb_results = nb_results.sort_values('Project').reset_index(drop=True)
bert_results = bert_results.sort_values('Project').reset_index(drop=True)

# Metrics to compare
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

def main():
   
    # Process each metric and output results
    print("Paired T-test Results")
    print("=" * 50)
    
    # First metric
    a_tstat, a_pval = stats.ttest_rel(nb_results['Accuracy'], bert_results['Accuracy'])
    print("Accuracy:")
    print(f"  t-statistic: {a_tstat:.4f}")
    print(f"  p-value: {a_pval:.5f}")
    if a_pval < 0.05:
        print("  BERT significantly outperforms NB (p < 0.05)")
    else:
        print("  No significant difference (p >= 0.05)")
    
    # Second metric
    print("\nPrecision:")
    p_tstat, p_pval = stats.ttest_rel(nb_results['Precision'], bert_results['Precision'])
    print(f"  t-statistic: {p_tstat:.4f}")
    print(f"  p-value: {p_pval:.4f}")
    if p_pval < 0.05:
        print("  BERT significantly outperforms NB (p < 0.05)")
    else:
        print("  No significant difference (p >= 0.05)")
    
    # Third metric
    print("\nRecall:")
    r_tstat, r_pval = stats.ttest_rel(nb_results['Recall'], bert_results['Recall'])
    print(f"  t-statistic: {r_tstat:.4f}")
    print(f"  p-value: {r_pval:.4f}")
    if r_pval < 0.05:
        print("  BERT significantly outperforms NB (p < 0.05)")
    else:
        print("  No significant difference (p >= 0.05)")
    
    # Fourth metric
    print("\nF1:")
    f_tstat, f_pval = stats.ttest_rel(nb_results['F1'], bert_results['F1'])
    print(f"  t-statistic: {f_tstat:.4f}")
    print(f"  p-value: {f_pval:.4f}")
    if f_pval < 0.05:
        print("  BERT significantly outperforms NB (p < 0.05)")
    else:
        print("  No significant difference (p >= 0.05)")

if __name__ == "__main__":
    main() 