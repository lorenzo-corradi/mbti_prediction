import streamlit as st
import pandas as pd
import re
import spacy
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from imageio import imread

class Exploration:
    
    df = {}
    
    # HELPER PARAMETERS
    __types = []
    __avg_words = 0
    __avg_http = 0
    __avg_question = 0
    __avg_exclamation = 0
    __avg_ellipsis = 0
    
    def __init__(self, df):
        self.df = df
                        
    # HELPER FUNCTION
    def __setHelperParameters(self):
        self.__types = sorted(self.df['type'].unique()) # sorted in alphabetical order to keep consistency with data
        self.__avg_words = round(self.df.groupby(['type'])['words'].apply(lambda x: np.mean(x)), 3)
        self.__avg_http = round(self.df.groupby(['type'])['http'].apply(lambda x: np.mean(x)), 3)
        self.__avg_question = round(self.df.groupby(['type'])['question'].apply(lambda x: np.mean(x)), 3)
        self.__avg_exclamation = round(self.df.groupby(['type'])['exclamation'].apply(lambda x: np.mean(x)), 3)
        self.__avg_ellipsis = round(self.df.groupby(['type'])['ellipsis'].apply(lambda x: np.mean(x)), 3)
                          
    
    def visualizeTargetVariable(self):
        
        self.__setHelperParameters()
        
        # PLOT DISTRIBUTION OF TARGET VARIABLE (df["type"])
        fig, ax = plt.subplots()
        sns.set_theme(style = "darkgrid")
        plt.suptitle("Distribution of target variable")
        ax1 = plt.subplot(111)
        sns.countplot(x = 'type', 
                      data = self.df, 
                      order = self.df["type"].value_counts(ascending = False).index)
        ax1.set_xticklabels(labels = self.df["type"].value_counts(ascending = False).index, rotation = 45)
        st.pyplot(fig)
        
    def visualizeWordsPerComment(self):
        
        self.__setHelperParameters()
        
        # PLOT (#WORDS / AVG #WORDS) PER COMMENT (PER TYPE)
        fig, ax = plt.subplots()
        fig.subplots_adjust(wspace = 0.4)
        sns.set_theme(style = "darkgrid")
        ax1 = plt.subplot(121)
        plt.title("Number of words per comment")
        sns.violinplot(x = 'type', 
                       y = 'words', 
                       data = self.df,
                       color = 'lightgray',
                       order = self.df["type"].value_counts(ascending = False).index)
        sns.stripplot(x = 'type', 
                      y = 'words', 
                      data = self.df, 
                      size = 4, 
                      jitter = True,
                      order = self.df["type"].value_counts(ascending = False).index)
        ax1.set_xticklabels(labels = self.df["type"].value_counts(ascending = False).index, rotation = 45)

        ax2 = plt.subplot(122)
        plt.title("Average number of words per comment")
        sns.barplot(x = self.__types, 
                    y = self.__avg_words)
        ax2.set_xticklabels(labels = self.__types, rotation = 45)

        st.pyplot(fig)
        
    def visualizeWordclouds(self):

        self.__setHelperParameters()
        
        # GENERATE WORDCLOUDS FOR EACH TYPE
        fig, axes = plt.subplots(nrows = int(self.df['type'].unique().shape[0] / 4), # hardcoded denominator
                                 ncols = int(self.df['type'].unique().shape[0] / 4), # hardcoded denominator
                                 # figsize = (18, 9)
                                 sharex = True)
        axes_flatten = axes.ravel()
        for i, ax in enumerate(axes_flatten):
            df_wordcloud = self.df[self.df['type'] == self.__types[i]]
            wordcloud = WordCloud(stopwords = STOPWORDS, 
                                  min_word_length = 2).generate(df_wordcloud['posts'].to_string())
            axes_flatten[i].imshow(wordcloud)
            axes_flatten[i].set_title(self.__types[i])
            axes_flatten[i].axis("off")
            
        st.pyplot(fig)

    def visualizeAverageStats(self):

        self.__setHelperParameters()

        # PLOT AVG: #HTTPs, #QUESTIONS, #EXCLAMATIONS, #ELLIPSIS PER COMMENT (PER TYPE)
        fig, ax = plt.subplots()
        fig.subplots_adjust(hspace = 0.6, wspace = 0.4)
        sns.set_theme(style = "darkgrid")

        ax1 = plt.subplot(221)
        plt.title("Links per comment")
        sns.barplot(x = self.__types,
                    y = self.__avg_http)
        ax1.set_xticklabels(labels = self.__types, rotation = 45)

        ax2 = plt.subplot(222)
        plt.title("Questions per comment")
        sns.barplot(x = self.__types, 
                    y = self.__avg_question)
        ax2.set_xticklabels(labels = self.__types, rotation = 45)

        ax3 = plt.subplot(223)
        plt.title("Exclamations per comment")
        sns.barplot(x = self.__types, 
                    y = self.__avg_exclamation)
        ax3.set_xticklabels(labels = self.__types, rotation = 45)

        ax4 = plt.subplot(224)
        plt.title("Ellipsis per comment")
        sns.barplot(x = self.__types, 
                    y = self.__avg_ellipsis)
        ax4.set_xticklabels(labels = self.__types, rotation = 45)
        
        st.pyplot(fig)