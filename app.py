import streamlit as st
import pandas as pd
from collections import Counter,OrderedDict
import utills
#import en_nlp_ner_transformer_pipeline
import streamlit_authenticator as stauth
from pathlib import Path
import pickle
import json
import warnings
from PIL import Image
import docx2txt
from io import StringIO 
import plotly.express as px


warnings.filterwarnings("ignore")

# --- page setting ---
img = Image.open("./data/icon.png")
st.set_page_config(
    page_title="AI CV Skill Clasifier",
    page_icon=img,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items= None
)
with open('style.css') as f:
        st.markdown(f'<style>{f.read()}<style>',unsafe_allow_html=True)

# --- user authentication ---
names = ["Md Alamgir Kabir", "Bo Telén Andersen", "ITOptimiser"]
usernames = ["mak","bta", "ITO"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {"usernames":{}}
        
for uname,name,pwd in zip(usernames,names,hashed_passwords):
    user_dict = {"name": name, "password": pwd}
    credentials["usernames"].update({uname: user_dict})

authenticator = stauth.Authenticate(credentials,"aicv_classifier", "abcdef", cookie_expiry_days=7)
name, authentication_status, username = authenticator.login("Login", "main")

if st.session_state["authentication_status"] == False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] == None:
    st.warning("Please enter your username and password")
elif st.session_state["authentication_status"]:
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    authenticator.logout("Logout", "sidebar")
    # Add a radio button widget to the sidebar
    selected_option = st.sidebar.radio(label = "Choose a processing option:", options=["Single File", "Multiple Files"]) 
    
    
    
    df_replace, df_search = utills.return_dataframes()
    replace_patterns = list(zip(df_replace["skill_name"], df_replace["replaced_by"]))
    search_patterns = list(zip(df_search["skill_name"], df_search["original_skill"]))
    categories = list(df_search.category.unique())

    # Load true skill dictionary
    with open('./data/TrueTargetDict.json') as f:
        trueTargets = json.loads(f.read())
    # functon to convert list to dictionary
    def lst_dict(lst):
        return {item: 0 for item in lst}
    # Title of the app
    st.title("AI CV Skill Classifier")
    st.subheader("Please Upload a SAP CV in .docx format")
    cont = st.container()
    if selected_option =='Single File':
        cvFile = cont.file_uploader("Select a CV :", type='.docx')
        button = cont.button("Analyze")
        
        if cvFile is not None and button:
            skillDict = utills.make_dataset_regex(cvFile, replace_patterns, search_patterns, multiple=False)
            if skillDict[list(skillDict.keys())[0]]:
                modelInputs, eachCategoryTotal = utills.divide_categories(skillDict, df_search, categories)
                finalBestOutput,selectedModelInput = utills.select_model_and_produce_results(modelInputs,df_search,trueTargets)
                if finalBestOutput[list(finalBestOutput.keys())[0]]["predSkills"][0] != 'No skill found!!':
                    plotSkills = OrderedDict(sorted(skillDict[list(skillDict.keys())[0]].items(), key=lambda x: x[1],reverse=True))
                    plotPieSkills = { k: v for k, v in plotSkills.items() if ( v > 1)} 
                    with st.container():
                        textCol, graphCol = st.columns([0.4, 0.6])
                        with textCol:
                            st.subheader('Analyzed Results:')
                            st.write("Record Id: ", list(skillDict.keys())[0])
                            st.write('Individual categories total :',eachCategoryTotal)
                            st.markdown(f'Max skills found in <span style="color:Blue">{list(modelInputs[list(modelInputs.keys())[0]].keys())[0]} </span> category', unsafe_allow_html=True)
                            st.write("Selected second Model:",list(modelInputs[list(modelInputs.keys())[0]].keys())[0])
                            st.write("Input to selected second model:", selectedModelInput)
                            #st.write("Skills found :", finalBestOutput)
                            st.write("Predicted skills by second model: ",", ".join(finalBestOutput[list(skillDict.keys())[0]]['predSkills']))
                            st.write("True skills: ",", ".join(finalBestOutput[list(skillDict.keys())[0]]['trunTarget']))
                            #st.write("True skills: ",", ".join(finalBestOutput[list(skillDict.keys())[0]]['trueTarget']))
                        with graphCol:
                            st.subheader('Visualized Results (First model output):')
                            #st.write("Skills list : ",skillDict)
                            # map primary skills to a color
                            color_map = dict(zip(modelInputs[list(modelInputs.keys())[0]][list(modelInputs[list(modelInputs.keys())[0]].keys())[0]].keys(), px.colors.qualitative.G10))
                            st.plotly_chart(utills.generate_bar_chart_plotly(plotSkills, 'All skills found on Record Id:  '+list(skillDict.keys())[0], color_map), use_container_width=True)
                            st.plotly_chart(utills.generate_pie_chart_plotly(modelInputs[list(modelInputs.keys())[0]], 'Pie Chart of primary skills for Record Id:  '+list(skillDict.keys())[0], color_map), use_container_width=True)
                else:
                    st.subheader('Analyzed Results:')
                    st.write("Record Id: ", list(finalBestOutput.keys())[0])
                    st.write('Predicted skills by second model: ',", ".join(finalBestOutput[list(finalBestOutput.keys())[0]]["predSkills"]))
                    st.write('True skills: ',", ".join(finalBestOutput[list(finalBestOutput.keys())[0]]['trunTarget']))
            else:
                st.write("No skill found in Record Id: ", list(skillDict.keys())[0])
        elif cvFile is None and button:
            st.warning('No file is choosen!! Please upload a file in .docx format', icon="⚠️")
        else:
            pass
           
    elif selected_option =='Multiple Files':
        display_option = st.sidebar.selectbox(label = "Choose result display option:", options=["Show all", "Only 100% Match", "Not 100% Match"])
        cvFiles = cont.file_uploader("Select CV(s) :", type='.docx', accept_multiple_files=True)
        button = cont.button("Analyze")
        
        if cvFiles and button:
            skillDict = utills.make_dataset_regex(cvFiles, replace_patterns, search_patterns, multiple=True)
            modelInputs,_ = utills.divide_categories(skillDict, df_search, categories)
            finalBestOutput,_ = utills.select_model_and_produce_results(modelInputs,df_search,trueTargets)
            st.subheader('Analyzed Results:')
            if display_option == "Show all":
                for recordId, result in finalBestOutput.items():
                    intersectedList = list(set(result['trunTarget']) & set(result['predSkills']))
                    formatedTarget = [f'<span class="matchedSkills">{skill}</span>' if skill in intersectedList else f'{skill}' for skill in result['trunTarget']]
                    formatedPredict = [f'<span class="matchedSkills">{skill}</span>' if skill in intersectedList else f'{skill}' for skill in result['predSkills']]
                    st.markdown(f'<h5>Record Id: {recordId}</h5>', unsafe_allow_html=True)
                    st.markdown('Predicted skills by second model: '+", ".join(formatedPredict), unsafe_allow_html=True)
                    st.markdown('True skills: '+", ".join(formatedTarget), unsafe_allow_html=True)
                    
            elif display_option == "Only 100% Match":
                for recordId, result in finalBestOutput.items():
                    if set(result['trunTarget']) == set(result['predSkills']):
                        st.markdown(f'<h5>Record Id: {recordId}</h5>', unsafe_allow_html=True)
                        st.write('Predicted skills by second model: ',", ".join(result['predSkills']))
                        st.write('True skills: ',", ".join(result['trunTarget']))
                    else:
                        pass
            else:
                for recordId, result in finalBestOutput.items():
                    if set(result['trunTarget']) != set(result['predSkills']):
                        intersectedList = list(set(result['trunTarget']) & set(result['predSkills']))
                        formatedTarget = [f'<span class="matchedSkills">{skill}</span>' if skill in intersectedList else f'{skill}' for skill in result['trunTarget']]
                        formatedPredict = [f'<span class="matchedSkills">{skill}</span>' if skill in intersectedList else f'{skill}' for skill in result['predSkills']]
                        st.markdown(f'<h5>Record Id: {recordId}</h5>', unsafe_allow_html=True)
                        st.markdown('Predicted skills by second model: '+", ".join(formatedPredict), unsafe_allow_html=True)
                        st.markdown('True skills: '+", ".join(formatedTarget), unsafe_allow_html=True)
                    else:
                        pass

        elif not cvFiles and button:
            st.warning('No file is choosen!! Please upload file(s) in .docx format', icon="⚠️")
        else:
            pass