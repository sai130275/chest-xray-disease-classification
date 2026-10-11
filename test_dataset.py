from dataset import load_data

df, mlb = load_data()

print("Images:", len(df))
print("Classes:", mlb.classes_)
print("Number of classes:", len(mlb.classes_))