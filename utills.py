# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 08:06:34 2022

@author: mak
"""
import pandas as pd
import numpy as np
import docx2txt
import re
from collections import Counter
import pickle
import bokeh.palettes as palette
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from math import pi
#import operator



def replace_matched_pattern_regex(pattern_keywords, content):
    """
    this functions takes two arguments as parameters
    - a list which contains a list of pattern to search and corresponding replacement keywords
    - document (candidates CV)
    """
    content = re.sub("\s+", " ", content)
    for items in pattern_keywords:
        try:
            content = re.sub(items[0], items[1], content)
        except:
            pass
    # removing of all special characters except "_"
    content = re.sub("[^A-Za-z0-9_]", " ", content)
    # removing extra white spaces
    content = re.sub("\s+", " ", content)
    return content

def make_dataset_regex_single_cv(fileName,pattern_keywords):
    content = docx2txt.process(fileName)
    content = replace_matched_pattern_regex(pattern_keywords,content)    
    all_skills=[]
    for item in pattern_keywords:
        allMatch = re.finditer(r"\b"+item[1]+r"\b", content)
        if(allMatch!=[]):
            for match in allMatch:
                all_skills.append(item[1])
    return [dict(Counter(all_skills))]


def createDataForSecondModelPrediction(data, skillList):
    finalDataDict = []
    try:
        item = {
            k: v for k, v in data.items() if (k in skillList)
        }
        if any(item):
            finalDataDict.append(item)
        else:
            pass
    except:
        pass
    return pd.DataFrame(finalDataDict, columns=skillList).fillna(0).values
    #return pd.DataFrame(finalDataDict, columns=skillList).fillna(0)
    
def get_hcm_model_output(data, originalHCMSkillList):
    with open('./models/HCMModelBest.pkl' , 'rb') as f:
        biModel = pickle.load(f)
        predict = biModel.predict(data)
        predictedSkillList = [" , ".join(originalHCMSkillList[row.astype(int).astype(np.bool)]) for row in predict]
    return predictedSkillList
def get_hybris_model_output(data, originalHybrisSkillList):
    with open('./models/HybrisModelBest.pkl' , 'rb') as f:
        biModel = pickle.load(f)
        predict = biModel.predict(data)
        predictedSkillList = [" , ".join(originalHybrisSkillList[row.astype(int).astype(np.bool)]) for row in predict]
    return predictedSkillList
def get_bi_model_output(data, originalBiSkillList):
    with open('./models/BiModelBest.pkl' , 'rb') as f:
        biModel = pickle.load(f)
        predict = biModel.predict(data)
        predictedSkillList = [" , ".join(originalBiSkillList[row.astype(int).astype(np.bool)]) for row in predict]
    return predictedSkillList
def get_fi_model_output(data, originalFiSkillList):
    with open('./models/FiModelBest.pkl' , 'rb') as f:
        fiModel = pickle.load(f)
        predict = fiModel.predict(data)
        predictedSkillList = [" , ".join(originalFiSkillList[row.astype(int).astype(np.bool)]) for row in predict]
    return predictedSkillList



def generate_figure(entities, titleText):
  color=palette.viridis(len(entities))
  p = figure(x_range = list(entities.keys()), title='', tools="hover", tooltips="@skills: @skill_count")
  source = ColumnDataSource(data=dict(skills = list(entities.keys()), skill_count = list(entities.values())))
  p.yaxis.axis_label = 'Counts'
  p.xaxis.axis_label = 'Skills'
  p.title.text = titleText
  p.plot_height=400
  p.xaxis.major_label_orientation = pi/3
  p.vbar(x = 'skills', top = 'skill_count', width = 0.8,source = source,  line_color='white', fill_color=factor_cmap('skills', palette=color, factors=list(entities.keys())))
  p.xgrid.grid_line_color = None
  p.y_range.start = 0.001
  return p

#def divide_categories(dataDict, df, categories, categoryThreshold = 0.7, truncatedValue=1):
def divide_categories(dataDict, df, categories, truncatedValue=1):
        finalDataDict = []
        for data in dataDict:
            catgoriesArray = []
            countEachCategory = []
            for catg in categories:
                try:
                    item = {
                        k: v for k, v in data.items() if (k in df[df.category == catg].replaced_by.values and v > truncatedValue)
                    }
                    if any(item):
                        catgoriesArray.append({catg: item})
                        countEachCategory.append(sum(item.values()))
                    else:
                        pass
                except:
                    pass
            '''percentage = [i/max(countEachCategory) for i in countEachCategory]
            indexes = [idx for idx, value in enumerate(percentage) if value >=categoryThreshold]
            finalDataDict.append(operator.itemgetter(*indexes)(catgoriesArray))
            finalDataDict.append(percentage)'''
            finalDataDict.append(catgoriesArray[np.argmax(countEachCategory)])
        return finalDataDict

def select_model_and_produce_results(modelInputs, df):
    finalBestOutputs =[]
    for data in modelInputs:
        if list(data.keys())[0] == 'BI Tools':
            inputForBiModel = createDataForSecondModelPrediction(data['BI Tools'], df[df.category == "BI Tools"].replaced_by.values)
            finalBestOutput = get_bi_model_output(inputForBiModel,df[df.category == "BI Tools"].original_skill.values)
        elif list(data.keys())[0] == 'Financial':
            inputForFiModel = createDataForSecondModelPrediction(data['Financial'], df[df.category == "Financial"].replaced_by.values)
            finalBestOutput = get_fi_model_output(inputForFiModel,df[df.category == "Financial"].original_skill.values)
        elif list(data.keys())[0] == 'HCM':
            inputForFiModel = createDataForSecondModelPrediction(data['HCM'], df[df.category == "HCM"].replaced_by.values)
            finalBestOutput = get_hcm_model_output(inputForFiModel,df[df.category == "HCM"].original_skill.values)
        elif list(data.keys())[0] == 'Hybris':
            inputForFiModel = createDataForSecondModelPrediction(data['Hybris'], df[df.category == "Hybris"].replaced_by.values)
            finalBestOutput = get_hybris_model_output(inputForFiModel,df[df.category == "Hybris"].original_skill.values)
        else:
            finalBestOutput = [f'{list(data.keys())[0]} category is not implemented yet!']
        finalBestOutputs.append(finalBestOutput)
    return finalBestOutputs
'''
def clean_data_for_second_model(dataPath, pattern_keywords):
    cleanData = []
    for fileName in os.listdir(dataPath):
        try:
            content = docx2txt.process(os.path.join(dataPath, fileName))
            # removing new line and tab spaces
            content = re.sub("\s+", " ", content)
            content = replace_matched_pattern(pattern_keywords, content)
            # removing of all special characters except "_"
            content = re.sub("[^A-Za-z0-9_]+", " ", content)
            # removing extra white spaces
            content = re.sub("\s+", " ", content)
            cleanData.append([content,{'fileIndex':fileName[:-5]}])
        except:
            pass
    return cleanData

def get_first_model_output(content):
    nlp = spacy.load('./models/model-best')
    nlp_doc = nlp(content)
    return [dict(Counter([ent.text for ent in nlp_doc.ents]))]


'''
    
    
    