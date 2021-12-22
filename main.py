from data_loader import DataLoader, RedditDataLoader, KaggleDataLoader
from exploration import Exploration
from preprocessing import Preprocessing
from learning import Learning

import streamlit as st

# def main():
    
    # USAGE
    
    # SCRAPE DATA FROM REDDIT
    # r_df = RedditDataLoader(20)
    # print(r_df.getDataFrame())
    
    # KAGGLE PRE-BUILT DATASET INITIALIZATION
    # k_df = KaggleDataLoader()
    # print(k_df.getDataFrame())
    
    
    # VISUALIZE STATISTICS
    
    # REDDIT
    # exp_r_df = r_df.addFeaturesForVisualization() # TAKES DataLoader OBJECT AS ARGUMENT, NOT A DataFrame
    # vis_r = Exploration(exp_r_df)
    # vis_r.visualizeTargetVariable()
    # vis_r.visualizeWordclouds()
    # vis_r.visualizeWordsPerComment()
    # vis_r.visualizeAverageStats()
    
    # KAGGLE
    # exp_k_df = k_df.addFeaturesForVisualization() # TAKES DataLoader OBJECT AS ARGUMENT, NOT A DataFrame
    # vis_k = Exploration(exp_k_df)
    # vis_k.visualizeTargetVariable()
    # vis_k.visualizeWordclouds()
    # vis_k.visualizeWordsPerComment()
    # vis_k.visualizeAverageStats()
    
    
    # TRAINING AND LEARNING

    # REDDIT
    
    
    # prep_r = Preprocessing(r_df.getDataFrame()) # getDataFrame() RETURNS A DataFrame
    # X_r, y_r, mbti_r = prep_r.preprocessing()
    # learn_r = Learning(X_r, y_r, mbti_r)
    # lambda_xgb_r = 0.4
    # learn_r.setXGBLambda(lambda_xgb_r) # NOT NECESSARY, DEFAULT: 0.3
    # learn_r.trainStratified(nsplits = 4)
    
    # MODEL USAGE
    # sentence = 'entp is gonna conquer world'
    # learn_r = Learning()
    # result_1, result_2 = learn_r.guessLabel([sentence], reddit = False)
    # print(result_1, result_2)
    
    # KAGGLE
    # prep_k = Preprocessing(k_df.getDataFrame()) # getDataFrame() RETURNS A DataFrame
    # X_k, y_k, mbti_k = prep_k.preprocessing()
    # learn_k = Learning(X_k, y_k, mbti_k)
    # lambda_xgb_k = 0.3
    # learn_k.setXGBLambda(lambda_xgb_k) # NOT NECESSARY, DEFAULT: 0.3
    # learn_k.trainStratified(nsplits = 5)
    
    # MODEL USAGE
    # sentence = 'debating is art'
    # learn_r = Learning()
    # result_1, result_2 = learn_r.guessLabel([sentence], reddit = False)
    # print(result_1, result_2)
    
    # return

@st.cache(allow_output_mutation = True)
def scrapeData(post_limit):
    r_df = RedditDataLoader(post_limit)
    return r_df

@st.cache(allow_output_mutation = True)
def preprocess(df):
    X, y, flair = Preprocessing(df).preprocessing()
    return X, y, flair
    
    
def main():
        
    # HERE STARTS STREAMLIT APP
    
    st.title('MBTI predictor')
    
    dataset = ['Kaggle', 'Reddit']
    vis = ['Target variable', 'Wordclouds', 'Stats on words', 'Stats on punctuation']
    r_df = RedditDataLoader(0)
    
    dataset_choice = st.sidebar.selectbox("Select dataset: ", dataset)

    if dataset_choice == 'Kaggle':
        st.subheader('Kaggle dataset')
        k_df = KaggleDataLoader()
        if st.button('Show dataset head'):
            k_head = k_df.getDataFrame()
            st.write(k_head.head())
        
    else:
        st.subheader('Reddit API')
        post_limit = st.slider('How many posts to scrape?', 0, 500, 30, 10)
        if (post_limit == 0):
            post_limit = 20
            st.write("At least scrape something! Selected " + str(post_limit) + " posts.")
        st.write("Please press the button below before choosing what to do with data.")
        if st.button("Start scraping"):
            r_df = scrapeData(post_limit)
            r_head = r_df.getDataFrame()
            st.write(r_head.head())
            st.write("Now choose graphic visualization or prediction.")
                
    if dataset_choice == 'Kaggle':
        st.write("Choose one of the following:")
        if st.checkbox("Start graphic visualization"):
            exp_k_df = k_df.addFeaturesForVisualization() # TAKES DataLoader OBJECT AS ARGUMENT, NOT A DataFrame
            vis_choice = st.selectbox('Select which stats to visualize:', vis)
            vis_k = Exploration(exp_k_df)
            if st.button("Show plot"):
                if vis_choice == 'Target variable':
                    vis_k.visualizeTargetVariable()
                elif vis_choice == 'Wordclouds':
                    vis_k.visualizeWordclouds()
                elif vis_choice == 'Stats on words':
                    vis_k.visualizeWordsPerComment()
                else:
                    vis_k.visualizeAverageStats()
    else:
        st.write("Choose one of the following:")
        if st.checkbox("Start graphic visualization"):
            exp_r_df = r_df.addFeaturesForVisualization() # TAKES DataLoader OBJECT AS ARGUMENT, NOT A DataFrame
            vis_choice = st.selectbox('Select which stats to visualize:', vis)
            vis_r = Exploration(exp_r_df)
            if st.button("Show plot"):
                if vis_choice == 'Target variable':
                    vis_r.visualizeTargetVariable()
                elif vis_choice == 'Wordclouds':
                    vis_r.visualizeWordclouds()
                elif vis_choice == 'Stats on words':
                    vis_r.visualizeWordsPerComment()
                else:
                    vis_r.visualizeAverageStats()

    if dataset_choice == 'Kaggle':
        if st.checkbox("Start prediction"):
            pred_k = Learning()
            sentence = st.text_input("Insert a sentence. Your MBTI will be based on this!")
            if st.button("Start prediction"):
                result_1, result_2 = pred_k.guessLabel([sentence], reddit = False)
                st.subheader("Results of the prediction: you are an " + result_1 + ", but you could also be an " + result_2 + "!")
    else:
        if st.checkbox("Start model training"):
            st.write("You chose to train the model with your retrieved data.")
            lambda_xgb_r = st.slider('Tune your lambda:', 0.0, 1.0, 0.3, 0.1)
            X_r, y_r, mbti_r = preprocess(r_df.getDataFrame())
            learn_r = Learning(X_r, y_r, mbti_r)
            learn_r.setXGBLambda(lambda_xgb_r)
            if st.button("Start training"):
                st.write("Please hold tight. This will take a good while!")
                learn_r.trainStratified()
                
        if st.checkbox("Start prediction"):
            sentence = st.text_input("Insert a sentence. Your MBTI will be based on this!")
            pred_r = Learning()
            if st.button("Start prediction"):
                result_1, result_2 = pred_r.guessLabel([sentence], reddit = True)
                st.subheader("Results of the prediction: you are an " + result_1 + ", but you could also be an " + result_2 + "!")
            
    return
        
if __name__ == "__main__":
    main()