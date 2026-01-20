import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Week 7: PRODUCTION Materials AI (SIMPLIFIED)")
print("=" * 70)

# Element properties
element_properties = {
    'Li': {'elec_neg': 0.98, 'ionic_rad': 0.76}, 'Ni': {'elec_neg': 1.91, 'ionic_rad': 0.69},
    'Co': {'elec_neg': 1.88, 'ionic_rad': 0.745}, 'Mn': {'elec_neg': 1.55, 'ionic_rad': 0.67},
    'O': {'elec_neg': 3.44, 'ionic_rad': 1.40}, 'Si': {'elec_neg': 1.90, 'ionic_rad': 0.40},
    'Ti': {'elec_neg': 1.54, 'ionic_rad': 0.86}, 'Al': {'elec_neg': 1.61, 'ionic_rad': 0.535}
}
elements = list(element_properties.keys())

# Dataset generator (Week 6)
class MaterialsDatasetRealistic:
    def __init__(self, elements):
        self.elements = elements
        self.techniques = ['XRD', 'TEM', 'XPS', 'Raman', 'PDF', 'SEM', 'ICP', 'NMR']
    
    def create_realistic_dataset(self, n_samples=20000):
        np.random.seed(123)
        compositions = np.random.dirichlet(np.ones(len(self.elements)), n_samples)
        battery_mask = np.random.random(n_samples) < 0.7
        compositions[battery_mask][:, [0,1,2,3,4]] *= 1.4
        compositions[~battery_mask][:, [4,5,6]] *= 1.4
        compositions = compositions / compositions.sum(axis=1, keepdims=True)
        
        rows = []
        for comp_vec in compositions:
            comp = {el: float(comp_vec[i] * (1 + np.random.normal(0, 0.12))) 
                   for i, el in enumerate(self.elements)}
            comp = {k: max(0, min(1, v)) for k, v in comp.items()}
            total = sum(comp.values())
            comp = {k: v/total for k, v in comp.items()}
            
            ni_co_mn = comp['Ni'] + comp['Co'] + comp['Mn']
            oxygen = comp['O']
            si_cont = comp['Si']
            
            techniques = []
            if np.random.random() < 0.92 and ni_co_mn > 0.38: techniques.append('XRD')
            if np.random.random() < 0.88 and ni_co_mn > 0.45: techniques.append('TEM')
            if np.random.random() < 0.90 and oxygen > 0.35: techniques.append('XPS')
            if np.random.random() < 0.85 and oxygen > 0.28: techniques.extend(['Raman', 'PDF'])
            if np.random.random() < 0.82 and si_cont > 0.18: techniques.append('SEM')
            if np.random.random() < 0.06: techniques.append('ICP')
            if np.random.random() < 0.04 and ni_co_mn > 0.6: techniques.append('NMR')
            if not techniques: techniques.append(np.random.choice(['XRD', 'SEM']))
            
            rows.append({'composition': comp, 'techniques': techniques,
                        'ni_co_mn': ni_co_mn, 'oxygen': oxygen, 'si_content': si_cont})
        return pd.DataFrame(rows)

print("Generating dataset...")
gen = MaterialsDatasetRealistic(elements)
df = gen.create_realistic_dataset(20000)

for el in elements:
    df[el] = df['composition'].apply(lambda d: d.get(el, 0.0))

# WEEK 7 FEATURES (14 total)
def advanced_features_v7(df):
    elec = np.array([element_properties[el]['elec_neg'] for el in elements])
    rad = np.array([element_properties[el]['ionic_rad'] for el in elements])
    
    df['avg_elec_neg'] = sum(df[el] * elec[i] for i, el in enumerate(elements))
    df['avg_ionic_rad'] = sum(df[el] * rad[i] for i, el in enumerate(elements))
    df['tm_ratio'] = df['ni_co_mn'] / (df['oxygen'] + 1e-8)
    df['si_oxygen_ratio'] = df['si_content'] / (df['oxygen'] + 1e-8)
    df['elec_neg_diff'] = df['avg_elec_neg'] - element_properties['O']['elec_neg']
    df['size_mismatch'] = sum(df[el] * (rad[i] - element_properties['O']['ionic_rad'])**2 
                             for i, el in enumerate(elements))
    
    # NEW FEATURES for ICP/NMR/SEM
    df['si_al_ratio'] = df['Si'] / (df['Al'] + 1e-8)
    df['si_tm_ratio'] = df['Si'] / (df[['Ni','Co','Mn']].sum(axis=1) + 1e-8)
    df['high_si_content'] = (df['si_content'] > 0.15).astype(float)
    df['element_diversity'] = df[elements].std(axis=1)
    df['tm_complexity'] = df[['Ni','Co','Mn']].std(axis=1)
    
    return df

df = advanced_features_v7(df)
feature_cols = elements + ['tm_ratio', 'si_oxygen_ratio', 'avg_elec_neg', 'avg_ionic_rad',
                          'elec_neg_diff', 'size_mismatch', 'ni_co_mn', 'oxygen', 'si_content',
                          'si_al_ratio', 'si_tm_ratio', 'high_si_content', 'element_diversity', 'tm_complexity']

# Encode + split
mlb = MultiLabelBinarizer(classes=gen.techniques)
y = mlb.fit_transform(df['techniques'])
X_tr, X_te, y_tr, y_te = train_test_split(df[feature_cols].fillna(0).values, y, test_size=0.2, random_state=42)

# WEEK 7: SINGLE ROBUST MODEL
print("Training Week 7 Production model...")
model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=1000,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features=0.3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    n_jobs=-1
)

model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

# Results
print("\n" + "="*70)
print("WEEK 7 PRODUCTION RESULTS")
print("="*70)
print(classification_report(y_te, y_pred, target_names=mlb.classes_, zero_division=0))
print(f"\n Subset Accuracy:  {accuracy_score(y_te, y_pred):.3%}")
print(f"Macro F1:         {f1_score(y_te, y_pred, average='macro'):.3f}")
print(f"Jaccard:          {jaccard_score(y_te, y_pred, average='samples'):.3f}")

# PRODUCTION API (SIMPLE + ROBUST)
class MaterialsAI_Production:
    def __init__(self, model, mlb, features, elements):
        self.model = model
        self.mlb = mlb
        self.features = features
        self.elements = elements
        self.techniques = list(mlb.classes_)
    
    def predict_techniques(self, composition, threshold=0.4):
        # Build feature vector
        feat = np.zeros(len(self.features))
        for i, el in enumerate(self.elements):
            feat[i] = composition.get(el, 0.0)
        
        # Compute physics features
        ni_co_mn = sum(composition.get(el, 0.0) for el in ['Ni', 'Co', 'Mn'])
        oxygen = composition.get('O', 0.0)
        si_cont = composition.get('Si', 0.0)
        al_cont = composition.get('Al', 0.0)
        
        elec = np.array([element_properties[el]['elec_neg'] for el in self.elements])
        rad = np.array([element_properties[el]['ionic_rad'] for el in self.elements])
        comp_vec = np.array([composition.get(el, 0.0) for el in self.elements])
        
        # Fill all features by index
        feat_names = self.features
        feat[feat_names.index('tm_ratio')] = ni_co_mn / (oxygen + 1e-8)
        feat[feat_names.index('si_oxygen_ratio')] = si_cont / (oxygen + 1e-8)
        feat[feat_names.index('avg_elec_neg')] = float((comp_vec * elec).sum())
        feat[feat_names.index('avg_ionic_rad')] = float((comp_vec * rad).sum())
        feat[feat_names.index('elec_neg_diff')] = feat[feat_names.index('avg_elec_neg')] - element_properties['O']['elec_neg']
        feat[feat_names.index('size_mismatch')] = float(((comp_vec * (rad - element_properties['O']['ionic_rad'])**2)).sum())
        feat[feat_names.index('ni_co_mn')] = ni_co_mn
        feat[feat_names.index('oxygen')] = oxygen
        feat[feat_names.index('si_content')] = si_cont
        feat[feat_names.index('si_al_ratio')] = si_cont / (al_cont + 1e-8)
        feat[feat_names.index('si_tm_ratio')] = si_cont / (ni_co_mn + 1e-8)
        feat[feat_names.index('high_si_content')] = 1.0 if si_cont > 0.15 else 0.0
        feat[feat_names.index('element_diversity')] = comp_vec.std()
        feat[feat_names.index('tm_complexity')] = np.std([composition.get(el, 0.0) for el in ['Ni','Co','Mn']])
        
        # Predict
        proba = model.predict_proba(feat.reshape(1, -1))
        probs = np.array([p[0, 1] for p in proba])
        techs = [self.techniques[i] for i, p in enumerate(probs) if p > threshold]
        
        return techs, probs

# Production demos
materials_ai = MaterialsAI_Production(model, mlb, feature_cols, elements)

print("\n" + "="*70)
print("PRODUCTION DEMOS")
print("="*70)

demos = {
    'NMC811': {'Li':0.167, 'Ni':0.333, 'Co':0.333, 'Mn':0.0, 'O':1.0},
    'LMO': {'Li':0.2, 'Mn':0.4, 'O':1.0},
    'Silicate': {'Si':0.35, 'O':1.0, 'Li':0.1},
    'TraceNi': {'Ni':0.01, 'O':1.0, 'Li':0.2, 'Si':0.1}
}

for name, comp in demos.items():
    # Fill missing elements with 0
    full_comp = {el: comp.get(el, 0.0) for el in elements}
    techs, probs = materials_ai.predict_techniques(full_comp)
    print(f"{name:8s}: {techs}")

# Save production pipeline
joblib.dump({
    'model': model,
    'mlb': mlb,
    'features': feature_cols,
    'elements': elements,
    'ai': materials_ai,
    'metadata': {
        'accuracy': float(accuracy_score(y_te, y_pred)),
        'macro_f1': float(f1_score(y_te, y_pred, average='macro')),
        'version': 'Week7_Production_v2.0'
    }
}, 'week7_production_ai.pkl')

print(f"\n WEEK 7 PRODUCTION COMPLETE!")
print(f"Accuracy: {accuracy_score(y_te, y_pred):.1%}")
print(f"Saved: week7_production_ai.pkl")