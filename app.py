import numpy as np
import pandas as pd
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from scripts.model import NNColabFiltering


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the model
    ratings = pd.read_csv('./data/ratings.csv')
    X = ratings.loc[:, ['userID', 'itemID']]
    n_users = X.loc[:, 'userID'].max()+1
    n_items = X.loc[:, 'itemID'].max()+1
    model = NNColabFiltering(n_users, n_items, embedding_dim_users=50,
                             embedding_dim_items=50, n_activations=100,
                             rating_range=[0., 10.])
    PATH = './models/model.pt'
    model.load_state_dict(torch.load(PATH, map_location=device))

    # Text cleaning
    profiles = pd.read_csv('./data/items.csv')
    profiles = profiles.rename(columns={'Unnamed: 0': 'itemID'})
    profiles.loc[:, 'essay0':'essay9'] = profiles.loc[:,
                                                      'essay0':'essay9'].fillna(' ')
    profiles['essay'] = profiles.loc[:,
                                     'essay0':'essay9'].apply(' '.join, axis=1)
    user_list = ratings['userID'].unique()
    user_list = sorted(user_list)
    # Get predicted rating for a specific user-item pair from model

    def predict_rating(model, userID, itemID, device):
        model = model.to(device)
        with torch.no_grad():
            model.eval()
            X = torch.Tensor([userID, itemID]).long().view(1, -1)
            X = X.to(device)
            pred = model.forward(X)
            return pred

    def generate_recommendations(profiles, model, userID, device):
        # Get predicted ratings for every movie
        pred_ratings = []
        for profile in profiles['itemID'].tolist():
            pred = predict_rating(model, userID, profile, device)
            pred_ratings.append(pred.detach().cpu().item())
        # Sort movies by predicted rating
        idxs = np.argsort(np.array(pred_ratings))[::-1]
        recs = profiles.iloc[idxs]
        return recs

    def generate_output(userID, top=5, sex='default', lower_age=0, upper_age=200, text=''):
        recs = generate_recommendations(profiles, model, userID, device)
        recs = recs[['itemID', 'sex', 'age', 'essay0', 'essay']]
        recs['sex'].replace({'f': 'female', 'm': 'male'}, inplace=True)

        # Applying age and gender filtering
        if sex == 'default':
            out = recs.loc[(recs['age'].between(lower_age, upper_age))]
        else:
            out = recs.loc[(recs['sex'] == sex) & (
                recs['age'].between(lower_age, upper_age))]

        out = out[:top]

        # Applying TF-IDF filtering
        if text != '':
            tfid = TfidfVectorizer(stop_words='english')
            matrix = tfid.fit_transform(out['essay'])
            text = tfid.transform([text])
            cosine_sim = cosine_similarity(matrix, text)
            cosine = pd.DataFrame(cosine_sim, columns=["similarity"])
            out = out.reset_index()
            out['match_for_input'] = cosine["similarity"]
            out = out.sort_values(by=['match_for_input'], ascending=False)
            out.drop(columns='essay', inplace=True)
            out.drop(columns='match_for_input', inplace=True)
            out.drop(columns='index', inplace=True)
            out.reset_index(drop=True, inplace=True)
            out.index += 1
            print("Sorted based on the input text")

        out.rename(columns={'itemID': 'profileID',
                   'essay0': 'self_intro'}, inplace=True)
        return out
 
 
    # UI
    
    tk = 0
    st.title('Dating App RecSys Microservice')

    Ages = list(range(18, 100))

    gender = ['female', 'male']
    topN = [5, 10, 20]
    gen = st.selectbox('Gender:', gender)
    userID = st.selectbox("User ID : ", user_list)
    top = st.selectbox('Top profiles:', topN)
    lower_age = st.selectbox("Minimum age : ", Ages)
    upper_age = st.selectbox("Maximum age : ", Ages)

    txt = st.text_input('Enter description : ')
    if st.button('Recommend'):
        tk = 1

    if tk == 1:
        st.success('Recommended profiles')
        st.empty()
        st.dataframe(generate_output(int(userID), int(
            top), gen, int(lower_age), int(upper_age), txt))


if __name__ == '__main__':
    main()
