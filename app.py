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
names = ["Md Alamgir Kabir", "Bo Telén Andersen"]
usernames = ["mak","bta"]

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
    df = pd.read_csv(
    "./data/more_than_15.csv",
    sep=";",
    encoding="latin1",
    usecols=["category", "regex_pattern", "replaced_by", "original_skill"],
)
    pattern_keywords = list(zip(df["regex_pattern"], df["replaced_by"], df["original_skill"]))
    categories = list(df.category.unique())

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
            skillDict = utills.make_dataset_regex(cvFile, pattern_keywords,multiple=False)
            if skillDict[list(skillDict.keys())[0]]:
                modelInputs = utills.divide_categories(skillDict, df, categories)
                finalBestOutput = utills.select_model_and_produce_results(modelInputs,df)
                plotSkills = OrderedDict(sorted(skillDict[list(skillDict.keys())[0]].items(), key=lambda x: x[1],reverse=True))
                with st.container():
                    textCol, graphCol = st.columns(2)
                    with textCol:
                        st.subheader('Analyzed Results:')
                        st.write("Record Id: ", list(skillDict.keys())[0])
                        #st.write(modelInputs)
                        st.markdown(f'Max skills found in <span style="color:Blue">{list(modelInputs[list(modelInputs.keys())[0]].keys())[0]} </span> category', unsafe_allow_html=True)
                        st.write("Selected second Model:",list(modelInputs[list(modelInputs.keys())[0]].keys())[0])
                        selectedSkillList = df[df.category == list(modelInputs[list(modelInputs.keys())[0]].keys())[0]].replaced_by.values
                        selectedSkillDict = lst_dict(selectedSkillList)
                        selectedSkillDict.update(modelInputs[list(modelInputs.keys())[0]][list(modelInputs[list(modelInputs.keys())[0]].keys())[0]])
                        st.write("Input to selected second model:", selectedSkillDict)
                        #st.write("Skills found :", finalBestOutput)
                        st.write("Predicted skills by second model: ",", ".join(finalBestOutput[list(skillDict.keys())[0]]))
                        trueTarget = trueTargets.get(list(skillDict.keys())[0])
                        if trueTarget:
                            st.write('True skills: ',trueTarget)
                        else:
                            st.write('True skill(s) is unknown.')
                    with graphCol:
                        st.subheader('Visualized Results (First model output):')
                        #st.write("Skills list : ",skillDict)
                        st.bokeh_chart(utills.generate_figure(plotSkills, 'All skills found on Record Id:  '+list(skillDict.keys())[0]), use_container_width=True)
            else:
                st.write("No skill found in Record Id: ", list(skillDict.keys())[0])
        elif cvFile is None and button:
            st.warning('No file is choosen!! Please upload a file in .docx format', icon="⚠️")
        else:
            pass
           
    elif selected_option =='Multiple Files':
        cvFiles = cont.file_uploader("Select CV(s) :", type='.docx', accept_multiple_files=True)
        button = cont.button("Analyze")
        
        if cvFiles and button:
            skillDict = utills.make_dataset_regex(cvFiles, pattern_keywords,multiple=True)
            modelInputs = utills.divide_categories(skillDict, df, categories)
            finalBestOutput = utills.select_model_and_produce_results(modelInputs,df)
            st.subheader('Analyzed Results:')
            for recordId, result in finalBestOutput.items():
                st.write("Record Id: ", recordId)
                st.write('Predicted skills by second model: ',", ".join(finalBestOutput[recordId]))
                trueTarget = trueTargets.get(recordId)
                if trueTarget:
                    st.write('True skills: ',trueTarget)
                else:
                    st.write('True skill(s) is unknown.')
        elif not cvFiles and button:
            st.warning('No file is choosen!! Please upload file(s) in .docx format', icon="⚠️")
        else:
            pass