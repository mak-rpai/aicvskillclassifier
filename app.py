import streamlit as st
import pandas as pd
from collections import Counter,OrderedDict
import utills
import en_nlp_ner_transformer_pipeline
import streamlit_authenticator as stauth
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

# --- page setting ---
st.set_page_config(
    page_title="AI CV Skill Clasifier",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
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

authenticator = stauth.Authenticate(credentials,"aicv_classifier", "abcdef", cookie_expiry_days=30)
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
elif authentication_status:
    st.sidebar.title(f"Welcome {name}")
    authenticator.logout("Logout", "sidebar")
    
    df = pd.read_csv(
    "./data/more_than_15.csv",
    sep=";",
    encoding="latin1",
    usecols=["category", "regex_pattern", "replaced_by", "original_skill"],
)
    pattern_keywords = list(zip(df["regex_pattern"], df["replaced_by"]))
    categories = list(df.category.unique())

    # Title of the app
    st.title("AI CV Skill Classifier")
    st.subheader("Please Upload a SAP CV in .docx format")
    cont = st.container()
    cvFile = cont.file_uploader("Select a CV :", type='.docx')
    button = cont.button("Analyze")
    @st.cache(allow_output_mutation=True)
    def get_nlp_model():
        nlp = en_nlp_ner_transformer_pipeline.load()
        return nlp

    nlp = get_nlp_model()

    if cvFile is not None and button:
        content = utills.clean_data_for_second_model_one_file(cvFile, pattern_keywords)
        skillDict = [dict(Counter([ent.text for ent in nlp(content).ents]))]
        modelInputs = utills.divide_categories(skillDict, df, categories)
        plotSkills = {
                    k: v for k, v in skillDict[0].items() if ( v > 1)
                }
        plotSkills = OrderedDict(sorted(plotSkills.items(), key=lambda x: x[1],reverse=True))
        finalBestOutput = utills.select_model_and_produce_results(modelInputs,df)
        with st.container():
            textCol, graphCol = st.columns(2)
            with textCol:
                st.subheader('Analyzed Results:')
                st.write("Record Id: ", cvFile.name[:-5])
                #st.write("Model Input:", modelInputs)
                #st.write("Skills found :", finalBestOutput)
                st.write("Skills found : ",", ".join(finalBestOutput[0]))
            with graphCol:
                st.subheader('Visualized Results:')
                #st.write("Skills list : ",skillDict)
                st.bokeh_chart(utills.generate_figure(plotSkills, 'Skill found on Record Id:  '+cvFile.name[:-5]), use_container_width=True)
    elif cvFile is None and button:
        st.warning('No file is choosen!! Please upload a file in .docx format', icon="⚠️")
    else:
        pass