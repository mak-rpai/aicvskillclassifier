import streamlit as st
import pandas as pd
from collections import Counter,OrderedDict
import utills
#import en_nlp_ner_transformer_pipeline
import streamlit_authenticator as stauth
from pathlib import Path
import pickle
import warnings
from PIL import Image

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
    
    df = pd.read_csv(
    "./data/more_than_15.csv",
    sep=";",
    encoding="latin1",
    usecols=["category", "regex_pattern", "replaced_by", "original_skill"],
)
    pattern_keywords = list(zip(df["regex_pattern"], df["replaced_by"], df["original_skill"]))
    categories = list(df.category.unique())
    # functon to convert list to dictionary
    def lst_dict(lst):
        return {item: 0 for item in lst}
    # Title of the app
    st.title("AI CV Skill Classifier")
    st.subheader("Please Upload a SAP CV in .docx format")
    cont = st.container()
    cvFile = cont.file_uploader("Select a CV :", type='.docx')
    button = cont.button("Analyze")
    
    if cvFile is not None and button:
        skillDict = utills.make_dataset_regex_single_cv(cvFile, pattern_keywords)
        if(bool(skillDict[0])):
            modelInputs = utills.divide_categories(skillDict, df, categories)
            #plotSkills = {k: v for k, v in skillDict[0].items() if ( v > 1)}
            plotSkills = OrderedDict(sorted(skillDict[0].items(), key=lambda x: x[1],reverse=True))
            finalBestOutput = utills.select_model_and_produce_results(modelInputs,df)
            with st.container():
                textCol, graphCol = st.columns(2)
                with textCol:
                    st.subheader('Analyzed Results:')
                    st.write("Record Id: ", cvFile.name[:-5])
                    st.markdown(f'Max skills found in <span style="color:Blue">{list(modelInputs[0].keys())[0]} </span> category', unsafe_allow_html=True)
                    st.write("Selected second Model:",list(modelInputs[0].keys())[0])
                    selectedSkillList = df[df.category == list(modelInputs[0].keys())[0]].replaced_by.values
                    selectedSkillDict = lst_dict(selectedSkillList)
                    selectedSkillDict.update(modelInputs[0][list(modelInputs[0].keys())[0]])
                    st.write("Input to selected second model:", selectedSkillDict)
                    #st.write("Skills found :", finalBestOutput)
                    st.write("Predicted skills by second model: ",", ".join(finalBestOutput[0]))
                with graphCol:
                    st.subheader('Visualized Results (First model output):')
                    #st.write("Skills list : ",skillDict)
                    st.bokeh_chart(utills.generate_figure(plotSkills, 'All skills found on Record Id:  '+cvFile.name[:-5]), use_container_width=True)
        else:
            st.write("No skill found Record Id: ", cvFile.name[:-5])
    elif cvFile is None and button:
        st.warning('No file is choosen!! Please upload a file in .docx format', icon="⚠️")
    else:
        pass