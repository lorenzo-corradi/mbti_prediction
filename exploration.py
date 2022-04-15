from data_loader import DataLoader
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

class Exploration:
    
            
    def plot_target_variable(self, y):
        # PLOT DISTRIBUTION OF TARGET VARIABLE
        fig, ax = plt.subplots()
        sns.set_theme(style = "darkgrid")
        plt.suptitle("Distribution of target variable")
        ax1 = plt.subplot(111)
        
        sns.countplot(x = y,
                      data = y,
                      order = y.value_counts(ascending = False).index
                      )
        ax1.set_xticklabels(labels = y.value_counts(ascending = False).index, rotation = 45)
        plt.show()
        # st.pyplot(fig)
        
    def plot_wordclouds(self, X, y): # rewrite without self

        # GENERATE WORDCLOUDS FOR EACH TYPE
        fig, axes = plt.subplots(nrows = int(y.unique().shape[0] / 4), # hardcoded denominator
                                 ncols = int(y.unique().shape[0] / 4), # hardcoded denominator
                                 sharex = True)
        axes_flatten = axes.ravel()
        unique_flair_list = list(set(y))
        for i, ax in enumerate(axes_flatten):
            # X = X[y == unique_flair_list[i]]
            wordcloud = WordCloud(stopwords = STOPWORDS,
                                  min_word_length = 2).generate(X.to_string())
            axes_flatten[i].imshow(wordcloud)
            axes_flatten[i].set_title(unique_flair_list[i])
            axes_flatten[i].axis("off")
            
        plt.show()
        # st.pyplot(fig)
        
                
if __name__ == "__main__":
    
    data_loader = DataLoader()
    X = data_loader.load_clean_data_pandas()
    Exploration().plot_wordclouds(X['body'], X['author_flair_text'])