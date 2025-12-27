import pandas as pd
from pathlib import Path

csv_path = Path("data/processed/train.csv")
df = pd.read_csv(csv_path)

print("Баланс до:")
print(df['label'].value_counts())

kosatka_count = (df['label'] == 'Косатка').sum()
print(f"\nКосаток было: {kosatka_count}")

if kosatka_count > 1000:
    kosatka_df = df[df['label'] == 'Косатка'].sample(n=1000, random_state=42)
    other_df = df[df['label'] != 'Косатка']
    df = pd.concat([kosatka_df, other_df], ignore_index=True)

print("\nБаланс после:")
print(df['label'].value_counts())

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(csv_path, index=False)

print(f"\nСохранено {len(df)} образцов в {csv_path}")
