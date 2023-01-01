# In[1]
import pickle

# In[2]
save_path = "saved_models/"
modes = [
    "binary",
    "tfidf",
    "count",
    "freq"
]
model_names = ["LGR_" + mode for mode in modes]

# %% 
selection = 3
model_path = save_path + model_names[selection] + ".sav"
tok_path = "tokenizer.sav"


# %% 
model = pickle.load(open(model_path, 'rb'))
tokenizer = pickle.load(open(tok_path, 'rb'))


# %%  
def rating_sentiment(text, mode_idx):
    score = ''
    wordvec = tokenizer.texts_to_matrix(text, modes[mode_idx])
    score = model.predict(wordvec)
    sentiment = "Negative" if score == 0 else "Positive"
    return sentiment

def run(mode_idx):
    txt = input("Enter your comment:")
    print("Text:", txt)
    return rating_sentiment([txt], mode_idx)

# %%
print("Model:", model_names[selection])
print("Result:", run(selection))
# %%
