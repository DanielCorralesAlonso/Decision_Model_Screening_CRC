


def preprocessing(df):
    try:
        df.drop(columns = ["Medication", "año_reco", "fpi", "Unnamed: 0"], inplace = True)
    except:
        df = df

    try:
        df.drop(columns = ["prov_trab"], inplace = True)
    except:
        df = df
    
    try:
        df = df[(df["Age"] != "1_very_young") & (df["Age"] != "6_elderly")].copy()
    except:
        df = df

    return df